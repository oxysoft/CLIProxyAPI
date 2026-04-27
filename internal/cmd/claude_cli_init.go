// Package cmd: claude_cli_init.go installs a stub auth record that activates
// the ClaudeCLIExecutor in the proxy.
//
// Unlike the OAuth-backed providers (DoClaudeLogin, DoCodexLogin, …), the
// claude-cli mode delegates *all* credential handling to the locally
// installed `claude` binary, which reads ~/.claude/.credentials.json on its
// own. The proxy never sees a token; it only needs to know that "yes, this
// host is allowed to spawn claude as a subprocess."
//
// Init validates two preconditions and then writes a synthetic auth file:
//
//  1. the `claude` binary (or $CLAUDE_CLI_BINARY override) is on PATH and
//     responds to `--version`. Without that, ClaudeCLIExecutor would fail
//     on every spawn.
//
//  2. ~/.claude/.credentials.json exists, indicating the user has logged in
//     with Claude Code at least once. We don't validate the token — CC will
//     refresh as needed — but a missing file means the user hasn't signed
//     in and any spawn will immediately fail.
//
// The auth file is small enough to write inline rather than going through
// the full AuthManager.Login pipeline (which exists to drive OAuth).
package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	log "github.com/sirupsen/logrus"
)

const claudeCLIDefaultAuthFile = "claude-cli-default.json"

// claudeCLIAuthRecord mirrors the JSON shape of cliproxy/auth.Auth as it is
// persisted to disk. We define a local struct rather than importing the
// runtime type to keep cmd's dependency surface minimal.
type claudeCLIAuthRecord struct {
	ID            string            `json:"id"`
	Provider      string            `json:"provider"`
	Label         string            `json:"label,omitempty"`
	Status        string            `json:"status"`
	StatusMessage string            `json:"status_message,omitempty"`
	Disabled      bool              `json:"disabled"`
	Unavailable   bool              `json:"unavailable"`
	Attributes    map[string]string `json:"attributes,omitempty"`
	Metadata      map[string]any    `json:"metadata,omitempty"`
	CreatedAt     time.Time         `json:"created_at"`
	UpdatedAt     time.Time         `json:"updated_at,omitempty"`
}

// DoClaudeCLIInit registers the claude-cli executor without any OAuth flow.
// It validates the local Claude Code installation and writes a stub auth
// record into cfg.AuthDir. Subsequent server starts will pick it up via
// the standard auth-loading machinery.
func DoClaudeCLIInit(cfg *config.Config, _ *LoginOptions) {
	if cfg == nil {
		log.Error("claude-cli init requires a loaded configuration")
		return
	}

	binary := os.Getenv("CLAUDE_CLI_BINARY")
	if strings.TrimSpace(binary) == "" {
		binary = "claude"
	}
	if err := verifyClaudeCLIBinary(binary); err != nil {
		log.Errorf("claude-cli init: %v", err)
		fmt.Println("Install Claude Code (https://docs.claude.com/en/docs/claude-code/setup) and try again.")
		fmt.Println("Or set CLAUDE_CLI_BINARY=/path/to/claude before re-running.")
		return
	}

	credsPath, err := claudeCLICredentialsPath()
	if err != nil {
		log.Errorf("claude-cli init: %v", err)
		return
	}
	if _, err := os.Stat(credsPath); err != nil {
		log.Errorf("claude-cli init: missing Claude Code credentials at %s: %v", credsPath, err)
		fmt.Printf("Run `%s /login` to sign in with your Claude account, then re-run --claude-cli-init.\n", binary)
		return
	}

	authDir := cfg.AuthDir
	if strings.TrimSpace(authDir) == "" {
		log.Error("claude-cli init: auth-dir is not set in config")
		return
	}
	if err := os.MkdirAll(authDir, 0o700); err != nil {
		log.Errorf("claude-cli init: create auth-dir %s: %v", authDir, err)
		return
	}

	authPath := filepath.Join(authDir, claudeCLIDefaultAuthFile)
	now := time.Now().UTC()
	record := claudeCLIAuthRecord{
		ID:       "claude-cli-" + uuid.NewString(),
		Provider: "claude-cli",
		Label:    "Local Claude Code CLI",
		Status:   "active",
		Attributes: map[string]string{
			"transport":          "subprocess",
			"binary":             binary,
			"credentials_source": credsPath,
		},
		Metadata: map[string]any{
			"installed_at":      now.Format(time.RFC3339),
			"claude_cli_version": claudeCLIVersionOrEmpty(binary),
		},
		CreatedAt: now,
		UpdatedAt: now,
	}

	body, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		log.Errorf("claude-cli init: marshal auth record: %v", err)
		return
	}
	if err := os.WriteFile(authPath, body, 0o600); err != nil {
		log.Errorf("claude-cli init: write %s: %v", authPath, err)
		return
	}

	fmt.Printf("Claude CLI executor registered.\n")
	fmt.Printf("  binary:        %s\n", binary)
	fmt.Printf("  credentials:   %s\n", credsPath)
	fmt.Printf("  auth record:   %s\n", authPath)
	fmt.Println()
	fmt.Println("Start the proxy with --config <yaml> as usual; the executor will pick this auth up on boot.")
}

func verifyClaudeCLIBinary(binary string) error {
	path, err := exec.LookPath(binary)
	if err != nil {
		return fmt.Errorf("claude binary %q not found on PATH: %w", binary, err)
	}
	cmd := exec.Command(path, "--version")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("claude --version failed: %w (output: %s)", err, strings.TrimSpace(string(out)))
	}
	return nil
}

func claudeCLIVersionOrEmpty(binary string) string {
	out, err := exec.Command(binary, "--version").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func claudeCLICredentialsPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	return filepath.Join(home, ".claude", ".credentials.json"), nil
}
