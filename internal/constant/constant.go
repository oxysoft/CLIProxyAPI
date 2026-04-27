// Package constant defines provider name constants used throughout the CLI Proxy API.
// These constants identify different AI service providers and their variants,
// ensuring consistent naming across the application.
package constant

const (
	// Gemini represents the Google Gemini provider identifier.
	Gemini = "gemini"

	// GeminiCLI represents the Google Gemini CLI provider identifier.
	GeminiCLI = "gemini-cli"

	// Codex represents the OpenAI Codex provider identifier.
	Codex = "codex"

	// Claude represents the Anthropic Claude provider identifier.
	Claude = "claude"

	// ClaudeCLI represents the Claude Code CLI subprocess provider identifier.
	// Mirrors the GeminiCLI pattern: same upstream model family, different
	// transport. Spawns the local `claude` binary in print mode and translates
	// its stream-json output to OpenAI Responses SSE. Subscription-billed
	// natively via the user's existing Claude Code OAuth credentials.
	ClaudeCLI = "claude-cli"

	// OpenAI represents the OpenAI provider identifier.
	OpenAI = "openai"

	// OpenaiResponse represents the OpenAI response format identifier.
	OpenaiResponse = "openai-response"

	// Antigravity represents the Antigravity response format identifier.
	Antigravity = "antigravity"
)
