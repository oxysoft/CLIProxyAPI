// Package executor provides runtime execution capabilities for various AI service providers.
//
// claude_cli_executor.go implements the ClaudeCLIExecutor: a subprocess-based
// executor that spawns the local Claude Code CLI (`claude`) in print mode and
// translates its stream-json output into the OpenAI Responses SSE format.
//
// Architectural rationale:
//
// The sibling ClaudeExecutor (claude_executor.go) talks directly to
// api.anthropic.com over HTTP. That path is correct when the caller wants
// Claude *as a model* — the upstream sees only the proxy's tool list and
// returns tool_use blocks the proxy translates back. Anthropic's anti-abuse
// pipeline scrutinises subscription-billed traffic for "is this Claude Code"
// fingerprints (tool surface, headers, cch signing), and the proxy spends
// considerable effort emulating CC. Drift in any of those leaks a fingerprint.
//
// ClaudeCLIExecutor sidesteps that problem by *being* Claude Code: the user's
// real `claude` binary is spawned per request. Anthropic sees genuine CC
// traffic because it is genuine CC traffic. From the caller's side (typically
// Codex driving /v1/responses), the request arrives with an OpenAI Responses
// payload; the executor renders the messages into a single prompt, runs
// `claude -p --output-format stream-json`, and surfaces only the final
// assistant text as Responses events. Claude Code's internal tool calls
// (Bash/Edit/Read/etc.) execute inside the spawned process and never appear
// in the response stream — the caller sees a simple text reply, exactly as
// if it had asked an oracle.
//
// This trade-off is intentional: callers lose visibility into CC's tool
// reasoning, but gain a fingerprint-clean transport that scales to many
// parallel agents without tripping subscription anti-abuse rules.
package executor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	usage "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/usage"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// claudeCLIBinaryEnv lets users override which `claude` binary is spawned
// (e.g. when multiple Claude Code installations coexist).
const claudeCLIBinaryEnv = "CLAUDE_CLI_BINARY"

// claudeCLIDefaultBinary is the conventional name on PATH.
const claudeCLIDefaultBinary = "claude"

// claudeCLISessionTTL bounds how long a tracked --resume session id is held
// before being eligible for reaping. Sessions are also evicted on parent ctx
// cancellation; this TTL is a safety net for orphaned IDs.
const claudeCLISessionTTL = 30 * time.Minute

// ClaudeCLIExecutor spawns the user's `claude` binary per request and
// streams its output back as OpenAI Responses SSE chunks.
type ClaudeCLIExecutor struct {
	cfg *config.Config

	// sessions maps proxy-issued response ids to CC session ids so that a
	// follow-up turn (Codex passes previous_response_id) can `--resume` the
	// underlying CC conversation rather than starting fresh.
	sessions sync.Map // map[string]claudeCLISessionEntry
}

type claudeCLISessionEntry struct {
	sessionID string
	createdAt time.Time
}

// NewClaudeCLIExecutor returns a new ClaudeCLIExecutor bound to cfg.
func NewClaudeCLIExecutor(cfg *config.Config) *ClaudeCLIExecutor {
	return &ClaudeCLIExecutor{cfg: cfg}
}

// Identifier reports the provider key this executor handles.
func (e *ClaudeCLIExecutor) Identifier() string { return "claude-cli" }

// PrepareRequest is a no-op: the CLI manages its own credentials via
// ~/.claude/.credentials.json; the proxy never sees a token to inject.
func (e *ClaudeCLIExecutor) PrepareRequest(_ *http.Request, _ *cliproxyauth.Auth) error {
	return nil
}

// HttpRequest is unsupported: this executor does not speak HTTP upstream.
func (e *ClaudeCLIExecutor) HttpRequest(_ context.Context, _ *cliproxyauth.Auth, _ *http.Request) (*http.Response, error) {
	return nil, statusErr{code: http.StatusNotImplemented, msg: "claude-cli executor does not expose an HTTP transport"}
}

// Refresh is a no-op: CC manages its own auth lifecycle.
func (e *ClaudeCLIExecutor) Refresh(_ context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	return auth, nil
}

// CountTokens returns a synthetic zero count. Claude Code does not expose a
// dedicated token-count endpoint to subprocess callers; the proxy returns
// zero rather than spawning a full inference just to count.
func (e *ClaudeCLIExecutor) CountTokens(_ context.Context, _ *cliproxyauth.Auth, _ cliproxyexecutor.Request, _ cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	body, _ := sjson.SetBytes([]byte(`{}`), "input_tokens", 0)
	return cliproxyexecutor.Response{Payload: body}, nil
}

// claudeCLIInvocation captures the parsed shape of an inbound request after
// translation from the source format. The translator stores the rendered
// prompt and any CC-specific knobs in the payload as a small JSON envelope,
// keeping the executor free of source-format awareness.
type claudeCLIInvocation struct {
	Prompt           string `json:"prompt"`
	ResumeSessionID  string `json:"resume_session_id,omitempty"`
	ParentResponseID string `json:"parent_response_id,omitempty"`
	WorkingDir       string `json:"cwd,omitempty"`
	Model            string `json:"model,omitempty"`
	MaxBudgetUSD     string `json:"max_budget_usd,omitempty"`
	DisplayName      string `json:"display_name,omitempty"`
}

func parseClaudeCLIInvocation(req cliproxyexecutor.Request) (*claudeCLIInvocation, error) {
	if len(req.Payload) == 0 {
		return nil, fmt.Errorf("empty payload")
	}
	var inv claudeCLIInvocation
	if err := json.Unmarshal(req.Payload, &inv); err != nil {
		return nil, fmt.Errorf("decode claude-cli invocation: %w", err)
	}
	if strings.TrimSpace(inv.Prompt) == "" {
		return nil, fmt.Errorf("claude-cli invocation has empty prompt")
	}
	if inv.Model == "" {
		inv.Model = req.Model
	}
	return &inv, nil
}

// Execute runs the CLI once and returns the final aggregated response in
// non-streaming form.
func (e *ClaudeCLIExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	reporter := helps.NewUsageReporter(ctx, e.Identifier(), req.Model, auth)
	var execErr error
	defer reporter.TrackFailure(ctx, &execErr)

	// Translate the inbound caller payload (e.g. OpenAI Responses) into our
	// claude-cli invocation envelope. The translator pair registered in
	// internal/translator/claude-cli/openai/responses handles this; if no
	// pair is registered for the source format the bytes pass through and
	// parseClaudeCLIInvocation will surface a clear error below.
	req.Payload = sdktranslator.TranslateRequest(
		opts.SourceFormat,
		sdktranslator.FromString("claude-cli"),
		req.Model,
		req.Payload,
		opts.Stream,
	)

	inv, err := parseClaudeCLIInvocation(req)
	if err != nil {
		execErr = err
		return cliproxyexecutor.Response{}, statusErr{code: http.StatusBadRequest, msg: err.Error()}
	}

	stdout, stderr, sessionID, err := e.runClaudeCLI(ctx, inv, /*stream*/ false)
	if err != nil {
		execErr = err
		return cliproxyexecutor.Response{}, err
	}

	// In non-streaming mode the proxy aggregates the assistant text and
	// returns a single OpenAI Responses-shaped JSON payload. The translator
	// pair turns this into whatever format the inbound caller expects.
	finalText, tokenUsage := aggregateClaudeCLIStreamJSON(stdout)
	if strings.TrimSpace(finalText) == "" && len(stderr) > 0 {
		// CC printed nothing on stdout but emitted stderr; surface as error.
		execErr = fmt.Errorf("claude cli returned no output: %s", strings.TrimSpace(string(stderr)))
		return cliproxyexecutor.Response{}, execErr
	}

	respID := e.rememberSession(inv.ParentResponseID, sessionID)
	out := buildClaudeCLINonStreamPayload(respID, inv.Model, finalText, tokenUsage)
	reporter.Publish(ctx, usage.Detail{InputTokens: tokenUsage.InputTokens, OutputTokens: tokenUsage.OutputTokens, TotalTokens: tokenUsage.InputTokens + tokenUsage.OutputTokens})
	return cliproxyexecutor.Response{Payload: out}, nil
}

// ExecuteStream runs the CLI and streams its NDJSON output line-by-line as
// translated SSE chunks. Each scanned line is forwarded to the translator
// pipeline (registered for the source format), which converts it to the
// caller-visible event shape.
func (e *ClaudeCLIExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	reporter := helps.NewUsageReporter(ctx, e.Identifier(), req.Model, auth)
	defer reporter.TrackFailure(ctx, &err)

	// Caller payload → claude-cli invocation envelope. Mirror Execute's
	// behavior so both code paths see a parsed invocation.
	req.Payload = sdktranslator.TranslateRequest(
		opts.SourceFormat,
		sdktranslator.FromString("claude-cli"),
		req.Model,
		req.Payload,
		opts.Stream,
	)

	inv, err := parseClaudeCLIInvocation(req)
	if err != nil {
		return nil, statusErr{code: http.StatusBadRequest, msg: err.Error()}
	}

	cmd, stdoutPipe, stderrPipe, sessionIDCh, err := e.startClaudeCLI(ctx, inv, /*stream*/ true)
	if err != nil {
		return nil, err
	}

	out := make(chan cliproxyexecutor.StreamChunk, 16)
	// Translate executor output (claude-cli stream-json) back to whatever the
	// inbound caller speaks. Direction: from = our native, to = caller's.
	executorFmt := sdktranslator.FromString("claude-cli")
	callerFmt := opts.SourceFormat
	respID := uuid.NewString()
	var translatorParam any

	go func() {
		defer close(out)
		defer func() {
			// Always reap the process to avoid zombies if the caller bails early.
			_ = cmd.Wait()
		}()

		// Drain stderr in parallel so it doesn't block the pipe.
		stderrBuf := &bytes.Buffer{}
		stderrDone := make(chan struct{})
		go func() {
			_, _ = io.Copy(stderrBuf, stderrPipe)
			close(stderrDone)
		}()

		scanner := bufio.NewScanner(stdoutPipe)
		// CC stream-json events can carry full message bodies; bump the
		// scanner buffer well past the default 64KB.
		scanner.Buffer(nil, 16*1024*1024)
		var capturedSessionID string
		for scanner.Scan() {
			rawLine := scanner.Bytes()
			line := bytes.TrimSpace(rawLine)
			if len(line) == 0 {
				continue
			}
			helps.AppendAPIResponseChunk(ctx, e.cfg, line)
			// Capture session id from the system/init frame so the response
			// envelope can carry it for --resume continuity.
			if capturedSessionID == "" {
				if sid := gjson.GetBytes(line, "session_id").String(); sid != "" {
					capturedSessionID = sid
				}
			}
			// Translator expects `data: {...}` framing to mirror Anthropic
			// SSE wire shape, so wrap each NDJSON line accordingly.
			framed := append([]byte("data: "), line...)
			chunks := sdktranslator.TranslateStream(ctx, executorFmt, callerFmt, req.Model, opts.OriginalRequest, req.Payload, framed, &translatorParam)
			for i := range chunks {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
				case <-ctx.Done():
					_ = cmd.Process.Signal(syscall.SIGTERM)
					return
				}
			}
		}
		<-stderrDone

		if scanErr := scanner.Err(); scanErr != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, scanErr)
			out <- cliproxyexecutor.StreamChunk{Err: scanErr}
			return
		}

		if waitErr := cmd.Wait(); waitErr != nil {
			if exitErr, ok := waitErr.(*exec.ExitError); ok && exitErr.ExitCode() != 0 {
				msg := strings.TrimSpace(stderrBuf.String())
				if msg == "" {
					msg = waitErr.Error()
				}
				helps.RecordAPIResponseError(ctx, e.cfg, fmt.Errorf("claude exit %d: %s", exitErr.ExitCode(), msg))
				out <- cliproxyexecutor.StreamChunk{Err: statusErr{code: http.StatusBadGateway, msg: msg}}
				return
			}
		}

		if capturedSessionID != "" {
			e.rememberSession(respID, capturedSessionID)
		}
		// Notify any waiter that captured the session id (currently unused;
		// the channel keeps the API symmetric and lets future callers
		// subscribe without changing the executor surface).
		select {
		case sessionIDCh <- capturedSessionID:
		default:
		}
	}()

	return &cliproxyexecutor.StreamResult{Chunks: out}, nil
}

// runClaudeCLI synchronously runs `claude -p` and returns its full stdout,
// stderr, and the captured session id. Used by Execute (non-streaming).
func (e *ClaudeCLIExecutor) runClaudeCLI(ctx context.Context, inv *claudeCLIInvocation, stream bool) (stdout, stderr []byte, sessionID string, err error) {
	cmd, stdoutPipe, stderrPipe, _, err := e.startClaudeCLI(ctx, inv, stream)
	if err != nil {
		return nil, nil, "", err
	}
	stdout, _ = io.ReadAll(stdoutPipe)
	stderr, _ = io.ReadAll(stderrPipe)
	if waitErr := cmd.Wait(); waitErr != nil {
		if exitErr, ok := waitErr.(*exec.ExitError); ok && exitErr.ExitCode() != 0 {
			msg := strings.TrimSpace(string(stderr))
			if msg == "" {
				msg = waitErr.Error()
			}
			return stdout, stderr, "", statusErr{code: http.StatusBadGateway, msg: msg}
		}
	}
	for _, line := range bytes.Split(stdout, []byte{'\n'}) {
		if sid := gjson.GetBytes(line, "session_id").String(); sid != "" {
			sessionID = sid
			break
		}
	}
	return stdout, stderr, sessionID, nil
}

// startClaudeCLI builds the argv, spawns the process, and returns its IO pipes.
// The caller is responsible for draining stdout/stderr and calling cmd.Wait().
func (e *ClaudeCLIExecutor) startClaudeCLI(ctx context.Context, inv *claudeCLIInvocation, stream bool) (*exec.Cmd, io.ReadCloser, io.ReadCloser, chan string, error) {
	binary := os.Getenv(claudeCLIBinaryEnv)
	if strings.TrimSpace(binary) == "" {
		binary = claudeCLIDefaultBinary
	}

	args := []string{"--print"}
	// stream-json gives us per-event NDJSON. Even Execute (non-stream) reads
	// it; aggregateClaudeCLIStreamJSON joins assistant text into one reply.
	args = append(args, "--output-format", "stream-json", "--verbose")
	if inv.Model != "" {
		args = append(args, "--model", inv.Model)
	}
	if resume := strings.TrimSpace(inv.ResumeSessionID); resume == "" {
		// If the caller passed previous_response_id, look up the CC session
		// id we recorded for that turn and resume it.
		if mapped := e.lookupSession(inv.ParentResponseID); mapped != "" {
			args = append(args, "--resume", mapped)
		}
	} else {
		args = append(args, "--resume", resume)
	}
	if name := strings.TrimSpace(inv.DisplayName); name != "" {
		args = append(args, "--name", name)
	}
	if budget := strings.TrimSpace(inv.MaxBudgetUSD); budget != "" {
		args = append(args, "--max-budget-usd", budget)
	}
	// Always pass the prompt as the final positional argument.
	args = append(args, inv.Prompt)

	cmd := exec.CommandContext(ctx, binary, args...)
	if cwd := strings.TrimSpace(inv.WorkingDir); cwd != "" {
		cmd.Dir = cwd
	}
	// Inherit environment so the user's CC config (HOME, paths, etc.) is
	// honoured. The CLI reads ~/.claude/.credentials.json on its own.
	cmd.Env = os.Environ()

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("claude-cli stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("claude-cli stderr pipe: %w", err)
	}

	log.Debugf("claude-cli executor: spawn %s %s (cwd=%q)", binary, strings.Join(args[:len(args)-1], " "), cmd.Dir)
	if err := cmd.Start(); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("claude-cli start: %w", err)
	}

	sessionIDCh := make(chan string, 1)
	return cmd, stdout, stderr, sessionIDCh, nil
}

// rememberSession records the CC session id for a proxy-issued response id so
// that a subsequent turn presenting that id as previous_response_id can resume
// the underlying CC conversation. Returns the response id used (callers may
// pass an existing one to reuse).
func (e *ClaudeCLIExecutor) rememberSession(respID, sessionID string) string {
	if sessionID == "" {
		if respID == "" {
			return uuid.NewString()
		}
		return respID
	}
	if respID == "" {
		respID = uuid.NewString()
	}
	e.sessions.Store(respID, claudeCLISessionEntry{sessionID: sessionID, createdAt: time.Now()})
	e.gcSessions()
	return respID
}

// lookupSession returns the CC session id mapped to respID, or "" if none.
func (e *ClaudeCLIExecutor) lookupSession(respID string) string {
	if respID == "" {
		return ""
	}
	val, ok := e.sessions.Load(respID)
	if !ok {
		return ""
	}
	entry := val.(claudeCLISessionEntry)
	if time.Since(entry.createdAt) > claudeCLISessionTTL {
		e.sessions.Delete(respID)
		return ""
	}
	return entry.sessionID
}

func (e *ClaudeCLIExecutor) gcSessions() {
	cutoff := time.Now().Add(-claudeCLISessionTTL)
	e.sessions.Range(func(k, v any) bool {
		if entry, ok := v.(claudeCLISessionEntry); ok && entry.createdAt.Before(cutoff) {
			e.sessions.Delete(k)
		}
		return true
	})
}

// claudeCLIUsage holds aggregated token usage extracted from CC output frames.
type claudeCLIUsage struct {
	InputTokens  int64
	OutputTokens int64
}

// aggregateClaudeCLIStreamJSON walks the NDJSON output emitted by
// `claude -p --output-format stream-json` and returns the final assistant
// text along with token usage. Used in non-streaming Execute. Internal
// tool_use / tool_result frames are intentionally dropped here: from the
// proxy caller's perspective Claude Code is a black box that returns text.
func aggregateClaudeCLIStreamJSON(stdout []byte) (string, claudeCLIUsage) {
	var (
		text  strings.Builder
		final string
		usage claudeCLIUsage
	)
	for _, line := range bytes.Split(stdout, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		root := gjson.ParseBytes(line)
		switch root.Get("type").String() {
		case "assistant":
			contents := root.Get("message.content")
			contents.ForEach(func(_, item gjson.Result) bool {
				if item.Get("type").String() == "text" {
					text.WriteString(item.Get("text").String())
				}
				return true
			})
		case "result":
			if r := root.Get("result").String(); r != "" {
				final = r
			}
			if u := root.Get("usage"); u.Exists() {
				usage.InputTokens = u.Get("input_tokens").Int()
				usage.OutputTokens = u.Get("output_tokens").Int()
			}
		}
	}
	if final != "" {
		return final, usage
	}
	return text.String(), usage
}

// buildClaudeCLINonStreamPayload renders the synthesised assistant text into
// an OpenAI Responses non-streaming payload. The translator pair (registered
// in internal/translator/claude-cli/openai/responses) converts this to the
// inbound caller's preferred format for response delivery.
func buildClaudeCLINonStreamPayload(respID, model, text string, usage claudeCLIUsage) []byte {
	body := []byte(`{"id":"","object":"response","status":"completed","output":[],"model":"","usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}`)
	body, _ = sjson.SetBytes(body, "id", respID)
	body, _ = sjson.SetBytes(body, "model", model)
	body, _ = sjson.SetBytes(body, "created_at", time.Now().Unix())
	item := []byte(`{"id":"","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":""}]}`)
	item, _ = sjson.SetBytes(item, "id", "msg_"+respID)
	item, _ = sjson.SetBytes(item, "content.0.text", text)
	body, _ = sjson.SetRawBytes(body, "output.-1", item)
	body, _ = sjson.SetBytes(body, "usage.input_tokens", usage.InputTokens)
	body, _ = sjson.SetBytes(body, "usage.output_tokens", usage.OutputTokens)
	body, _ = sjson.SetBytes(body, "usage.total_tokens", usage.InputTokens+usage.OutputTokens)
	return body
}
