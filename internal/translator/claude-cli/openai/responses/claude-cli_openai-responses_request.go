// Package responses (request side): OpenAI Responses → ClaudeCLI invocation.
//
// The Responses API request carries a structured `input` array of message
// objects (each with role + content blocks), plus model, instructions, and
// optional metadata. The Claude Code CLI in print mode accepts a single
// prompt argument; multi-turn continuity is achieved by --resume of a prior
// session id rather than by replaying the conversation in the prompt.
//
// This translator therefore renders the *latest user turn* into a single
// prompt string. Earlier conversation history is intentionally dropped from
// the prompt: if the caller has set previous_response_id, the executor
// looks up the recorded CC session id and resumes that session, which
// already holds the prior turns server-side. Re-replaying them in the
// prompt would double-bill and confuse CC's session state.
//
// `instructions` (the Responses-API analogue of a system prompt) is
// prepended verbatim. Multimodal content is collapsed to its text parts
// only; image inputs would need a separate path (CC's --print mode does not
// currently take inline image attachments).
package responses

import (
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertOpenAIResponsesRequestToClaudeCLI translates a Responses API
// request body into the small invocation envelope ClaudeCLIExecutor expects.
//
// Output schema (consumed by parseClaudeCLIInvocation):
//
//	{
//	  "prompt":             "<rendered prompt>",
//	  "model":              "<model id>",
//	  "parent_response_id": "<previous_response_id, if any>",
//	  "cwd":                "<metadata.cwd, if any>",
//	  "max_budget_usd":     "<metadata.max_budget_usd, if any>",
//	  "display_name":       "<metadata.display_name, if any>"
//	}
//
// `stream` is unused: the CLI always runs in stream-json mode internally;
// the executor decides whether to surface chunks or aggregate.
func ConvertOpenAIResponsesRequestToClaudeCLI(model string, rawJSON []byte, _ bool) []byte {
	envelope := []byte(`{}`)
	if model != "" {
		envelope, _ = sjson.SetBytes(envelope, "model", model)
	}

	root := gjson.ParseBytes(rawJSON)

	// previous_response_id signals a continuation; the executor maps this
	// to a recorded CC session id and passes --resume.
	if v := root.Get("previous_response_id"); v.Exists() {
		envelope, _ = sjson.SetBytes(envelope, "parent_response_id", v.String())
	}

	// metadata fields the caller may set to steer the spawned CLI.
	if v := root.Get("metadata.cwd"); v.Exists() {
		envelope, _ = sjson.SetBytes(envelope, "cwd", v.String())
	}
	if v := root.Get("metadata.max_budget_usd"); v.Exists() {
		envelope, _ = sjson.SetBytes(envelope, "max_budget_usd", v.String())
	}
	if v := root.Get("metadata.display_name"); v.Exists() {
		envelope, _ = sjson.SetBytes(envelope, "display_name", v.String())
	}

	prompt := renderResponsesInputAsPrompt(root)
	envelope, _ = sjson.SetBytes(envelope, "prompt", prompt)
	return envelope
}

// renderResponsesInputAsPrompt collapses the Responses request into a single
// string consumable by `claude -p`. It joins:
//
//  1. the top-level `instructions` field, if present (as a leading block);
//  2. the *latest* user turn from `input`.
//
// Earlier turns are dropped intentionally — see package doc comment.
func renderResponsesInputAsPrompt(root gjson.Result) string {
	var sections []string

	if instr := strings.TrimSpace(root.Get("instructions").String()); instr != "" {
		sections = append(sections, instr)
	}

	input := root.Get("input")
	if !input.Exists() {
		return strings.Join(sections, "\n\n")
	}

	// `input` may be a plain string (legacy short form) or an array of
	// message objects. Handle both.
	if input.Type == gjson.String {
		if s := strings.TrimSpace(input.String()); s != "" {
			sections = append(sections, s)
		}
		return strings.Join(sections, "\n\n")
	}

	// Walk the array and pick the last user message. If there is no user
	// message, fall back to concatenating all text content in order — this
	// preserves an obvious failure mode (caller sees their full input
	// echoed) rather than silently dropping the request.
	var lastUserText string
	var allText strings.Builder
	input.ForEach(func(_, msg gjson.Result) bool {
		role := msg.Get("role").String()
		text := extractResponsesMessageText(msg)
		if text == "" {
			return true
		}
		if allText.Len() > 0 {
			allText.WriteString("\n\n")
		}
		allText.WriteString(text)
		if role == "user" {
			lastUserText = text
		}
		return true
	})
	if lastUserText != "" {
		sections = append(sections, lastUserText)
	} else if allText.Len() > 0 {
		sections = append(sections, allText.String())
	}
	return strings.Join(sections, "\n\n")
}

// extractResponsesMessageText reads the text content out of a single
// Responses input-message object. The Responses API allows `content` to be
// either a string or a structured array of typed parts; both are handled.
// Non-text parts (images, audio) are silently dropped.
func extractResponsesMessageText(msg gjson.Result) string {
	content := msg.Get("content")
	if !content.Exists() {
		return ""
	}
	if content.Type == gjson.String {
		return strings.TrimSpace(content.String())
	}
	if !content.IsArray() {
		return ""
	}
	var b strings.Builder
	content.ForEach(func(_, part gjson.Result) bool {
		switch part.Get("type").String() {
		case "input_text", "output_text", "text":
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(part.Get("text").String())
		}
		return true
	})
	return strings.TrimSpace(b.String())
}
