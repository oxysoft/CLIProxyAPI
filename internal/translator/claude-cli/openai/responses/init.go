// Package responses registers translators between the OpenAI Responses
// wire format and the ClaudeCLIExecutor's native format.
//
// The pair is asymmetric:
//
//   - Request:  OpenAI Responses → ClaudeCLI invocation envelope
//     (a small JSON shape carrying the rendered prompt, optional
//     resume session id, working dir, and model. The executor parses
//     this directly via parseClaudeCLIInvocation.)
//
//   - Response (stream): each line of `claude -p --output-format
//     stream-json` (NDJSON, framed by the executor as `data: {...}`)
//     is converted to OpenAI Responses SSE events. CC's internal
//     tool_use / tool_result frames are intentionally dropped; only
//     final assistant text and the terminal `result` event surface
//     to the caller.
//
//   - Response (non-stream): the executor synthesises an aggregated
//     OpenAI Responses-shaped JSON envelope itself, so the non-stream
//     transform here is effectively a passthrough that hands the
//     payload back to the caller as-is.
package responses

import (
	. "github.com/router-for-me/CLIProxyAPI/v6/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/translator"
)

func init() {
	translator.Register(
		OpenaiResponse,
		ClaudeCLI,
		ConvertOpenAIResponsesRequestToClaudeCLI,
		interfaces.TranslateResponse{
			Stream:    ConvertClaudeCLIResponseToOpenAIResponses,
			NonStream: ConvertClaudeCLIResponseToOpenAIResponsesNonStream,
		},
	)
}
