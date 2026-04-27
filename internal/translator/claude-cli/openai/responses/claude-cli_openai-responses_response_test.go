package responses

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// TestClaudeCLIResponseTranslation_BasicSuccess feeds a minimal CC stream-json
// transcript through the translator and asserts the emitted OpenAI Responses
// SSE events carry the assistant text and a terminal completed event.
func TestClaudeCLIResponseTranslation_BasicSuccess(t *testing.T) {
	transcript := []string{
		`data: {"type":"system","subtype":"init","session_id":"sess-abc","model":"claude-opus-4-7"}`,
		`data: {"type":"assistant","message":{"id":"msg_1","role":"assistant","content":[{"type":"text","text":"Hello "}]},"session_id":"sess-abc"}`,
		`data: {"type":"assistant","message":{"id":"msg_1","role":"assistant","content":[{"type":"text","text":"world"}]},"session_id":"sess-abc"}`,
		`data: {"type":"result","subtype":"success","result":"Hello world","session_id":"sess-abc","usage":{"input_tokens":12,"output_tokens":3}}`,
	}

	var param any
	var allChunks [][]byte
	for _, line := range transcript {
		chunks := ConvertClaudeCLIResponseToOpenAIResponses(context.Background(), "claude-opus-4-7", nil, nil, []byte(line), &param)
		allChunks = append(allChunks, chunks...)
	}

	checked := map[string]bool{
		"response.created":           false,
		"response.in_progress":       false,
		"response.output_item.added": false,
		"response.output_text.delta": false,
		"response.output_text.done":  false,
		"response.output_item.done":  false,
		"response.completed":         false,
	}
	deltaCount := 0
	for _, chunk := range allChunks {
		event, payload := splitFrame(chunk)
		if event == "" {
			continue
		}
		checked[event] = true
		switch event {
		case "response.output_text.delta":
			deltaCount++
		case "response.output_text.done":
			if got := gjson.Get(payload, "text").String(); got != "Hello world" {
				t.Fatalf("output_text.done.text = %q, want %q", got, "Hello world")
			}
		case "response.output_item.done":
			if got := gjson.Get(payload, "item.content.0.text").String(); got != "Hello world" {
				t.Fatalf("output_item.done text = %q, want %q", got, "Hello world")
			}
			if got := gjson.Get(payload, "item.type").String(); got != "message" {
				t.Fatalf("output_item.done.item.type = %q, want %q", got, "message")
			}
			if got := gjson.Get(payload, "item.role").String(); got != "assistant" {
				t.Fatalf("output_item.done.item.role = %q, want %q", got, "assistant")
			}
		case "response.completed":
			if got := gjson.Get(payload, "response.status").String(); got != "completed" {
				t.Fatalf("response.completed.status = %q, want completed", got)
			}
			if got := gjson.Get(payload, "response.usage.input_tokens").Int(); got != 12 {
				t.Fatalf("response.completed.usage.input_tokens = %d, want 12", got)
			}
			if got := gjson.Get(payload, "response.usage.output_tokens").Int(); got != 3 {
				t.Fatalf("response.completed.usage.output_tokens = %d, want 3", got)
			}
		}
	}
	for evt, seen := range checked {
		if !seen {
			t.Fatalf("expected event %q in stream output (saw %d chunks)", evt, len(allChunks))
		}
	}
	if deltaCount != 2 {
		t.Fatalf("expected 2 output_text.delta events (one per assistant content frame), got %d", deltaCount)
	}
}

// TestClaudeCLIResponseTranslation_DropsInternalToolFrames verifies that CC's
// internal tool_use / tool_result events do not surface as Responses-API
// chunks. From the proxy caller's perspective Claude Code is a black box.
func TestClaudeCLIResponseTranslation_DropsInternalToolFrames(t *testing.T) {
	transcript := []string{
		`data: {"type":"system","subtype":"init","session_id":"sess-1","model":"claude-opus-4-7"}`,
		`data: {"type":"assistant","message":{"id":"msg_1","role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"/tmp/x"}}]},"session_id":"sess-1"}`,
		`data: {"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"contents"}]},"session_id":"sess-1"}`,
		`data: {"type":"assistant","message":{"id":"msg_1","role":"assistant","content":[{"type":"text","text":"final answer"}]},"session_id":"sess-1"}`,
		`data: {"type":"result","subtype":"success","result":"final answer","session_id":"sess-1","usage":{"input_tokens":1,"output_tokens":1}}`,
	}

	var param any
	var deltas []string
	var sawInternalLeak bool
	for _, line := range transcript {
		chunks := ConvertClaudeCLIResponseToOpenAIResponses(context.Background(), "claude-opus-4-7", nil, nil, []byte(line), &param)
		for _, chunk := range chunks {
			event, payload := splitFrame(chunk)
			if event == "response.output_text.delta" {
				deltas = append(deltas, gjson.Get(payload, "delta").String())
			}
			if strings.Contains(payload, "tool_use") || strings.Contains(payload, "tool_result") {
				sawInternalLeak = true
			}
		}
	}
	if sawInternalLeak {
		t.Fatalf("internal tool frames must not surface in Responses output; saw a leak")
	}
	if len(deltas) != 1 || deltas[0] != "final answer" {
		t.Fatalf("expected single delta %q, got %v", "final answer", deltas)
	}
}

// TestClaudeCLIResponseTranslation_PartialMessagesStreamEvents verifies that
// when CC is invoked with --include-partial-messages, the per-token
// `stream_event` frames are surfaced as response.output_text.delta events
// (one per token) and the redundant cumulative `assistant` frame is skipped
// rather than double-emitting the assembled text.
func TestClaudeCLIResponseTranslation_PartialMessagesStreamEvents(t *testing.T) {
	transcript := []string{
		`data: {"type":"system","subtype":"init","session_id":"sess-2","model":"claude-opus-4-7"}`,
		`data: {"type":"stream_event","event":{"type":"message_start","message":{"id":"msg_partial","role":"assistant","content":[]}},"session_id":"sess-2"}`,
		`data: {"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}},"session_id":"sess-2"}`,
		`data: {"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}},"session_id":"sess-2"}`,
		`data: {"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}},"session_id":"sess-2"}`,
		`data: {"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}},"session_id":"sess-2"}`,
		`data: {"type":"stream_event","event":{"type":"content_block_stop","index":0},"session_id":"sess-2"}`,
		// Cumulative recap from CC — must be SKIPPED because we already streamed.
		`data: {"type":"assistant","message":{"id":"msg_partial","role":"assistant","content":[{"type":"text","text":"Hello!"}]},"session_id":"sess-2"}`,
		`data: {"type":"stream_event","event":{"type":"message_stop"},"session_id":"sess-2"}`,
		`data: {"type":"result","subtype":"success","result":"Hello!","session_id":"sess-2","usage":{"input_tokens":2,"output_tokens":3}}`,
	}

	var param any
	var allChunks [][]byte
	for _, line := range transcript {
		chunks := ConvertClaudeCLIResponseToOpenAIResponses(context.Background(), "claude-opus-4-7", nil, nil, []byte(line), &param)
		allChunks = append(allChunks, chunks...)
	}

	// Concatenate all output_text.delta payloads — should reconstruct "Hello!"
	// exactly once (3 token chunks). If the assistant recap leaked through, we
	// would see "Hello!Hello!" or similar duplication.
	var deltas []string
	var sawCompleted bool
	for _, chunk := range allChunks {
		event, payload := splitFrame(chunk)
		switch event {
		case "response.output_text.delta":
			deltas = append(deltas, gjson.Get(payload, "delta").String())
		case "response.completed":
			sawCompleted = true
			if got := gjson.Get(payload, "response.usage.output_tokens").Int(); got != 3 {
				t.Fatalf("response.completed.usage.output_tokens = %d, want 3", got)
			}
		}
	}
	if got := strings.Join(deltas, ""); got != "Hello!" {
		t.Fatalf("concatenated deltas = %q, want %q (chunks=%d)", got, "Hello!", len(deltas))
	}
	if len(deltas) != 3 {
		t.Fatalf("expected 3 token-level deltas, got %d", len(deltas))
	}
	if !sawCompleted {
		t.Fatalf("expected response.completed frame, did not see one")
	}
}

func splitFrame(chunk []byte) (event, payload string) {
	for _, line := range strings.Split(string(chunk), "\n") {
		switch {
		case strings.HasPrefix(line, "event: "):
			event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			payload = strings.TrimPrefix(line, "data: ")
		}
	}
	return event, payload
}
