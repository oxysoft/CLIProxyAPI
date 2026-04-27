// Package responses (response side): ClaudeCLI stream-json → OpenAI Responses.
//
// Claude Code's `--print --output-format stream-json --verbose` mode emits a
// stream of NDJSON event objects. The shapes that matter for translation:
//
//	{"type":"system","subtype":"init","session_id":"...", ...}
//	{"type":"assistant","message":{"id":"msg_...","role":"assistant",
//	    "content":[{"type":"text","text":"..."}, ...]}, "session_id":"..."}
//	{"type":"user","message":{...,"content":[{"type":"tool_result",...}]}}
//	{"type":"result","subtype":"success","result":"...","usage":{...}}
//
// `assistant` frames may carry text *or* tool_use blocks (Bash, Edit, …).
// `user` frames here are CC's internal echo of tool_result blocks. Both
// internal-tool frames are dropped: from the Responses-API caller's
// perspective Claude Code is a single-turn oracle that returns text. The
// final `result` frame carries the consolidated assistant text and usage.
//
// This translator therefore emits, in SSE order:
//
//	response.created
//	response.in_progress
//	response.output_item.added           (assistant message item, empty)
//	response.content_part.added          (output_text part, empty)
//	response.output_text.delta           (one per assistant text delta)
//	response.output_text.done            (replays accumulated text)
//	response.content_part.done
//	response.output_item.done
//	response.completed
//
// The shape mirrors what the Anthropic-Messages → OpenAI-Responses
// translator emits (see internal/translator/claude/openai/responses), so
// downstream Responses consumers see a consistent event stream regardless
// of which Claude transport produced it.
package responses

import (
	"context"
	"strings"
	"time"

	translatorcommon "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/common"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// claudeCLIResponseState carries per-stream bookkeeping across consecutive
// invocations of ConvertClaudeCLIResponseToOpenAIResponses. The runtime
// stores the value through the `*param` pointer so that successive lines
// from the same response share state.
type claudeCLIResponseState struct {
	seq        int
	responseID string
	createdAt  int64
	itemID     string
	textBuf    strings.Builder
	openedItem bool
	closedItem bool
	completed  bool
}

var dataTag = []byte("data:")

func emit(event string, payload []byte) []byte {
	return translatorcommon.SSEEventData(event, payload)
}

// ConvertClaudeCLIResponseToOpenAIResponses converts a single CC stream-json
// line (framed by the executor as `data: {...}`) into zero or more OpenAI
// Responses SSE chunks. State persists across invocations through *param.
func ConvertClaudeCLIResponseToOpenAIResponses(_ context.Context, _ string, _, _, rawJSON []byte, param *any) [][]byte {
	if *param == nil {
		*param = &claudeCLIResponseState{}
	}
	st := (*param).(*claudeCLIResponseState)
	nextSeq := func() int { st.seq++; return st.seq }

	if !startsWithDataTag(rawJSON) {
		return nil
	}
	body := trimDataTag(rawJSON)
	root := gjson.ParseBytes(body)
	if !root.Exists() {
		return nil
	}

	var out [][]byte

	switch root.Get("type").String() {
	case "system":
		// Init frame carries session_id + model. Use it to seed the response
		// envelope; subsequent assistant frames will reference st.responseID.
		if root.Get("subtype").String() != "init" {
			return nil
		}
		if st.responseID == "" {
			if sid := root.Get("session_id").String(); sid != "" {
				st.responseID = sid
			} else {
				st.responseID = "resp_" + nowSuffix()
			}
		}
		if st.createdAt == 0 {
			st.createdAt = time.Now().Unix()
		}
		out = append(out, emitCreated(st, nextSeq))
		out = append(out, emitInProgress(st, nextSeq))

	case "assistant":
		// Open the message item lazily, on the first assistant frame, so
		// short prompts that produce no output (CC error before reply) don't
		// emit a half-formed envelope.
		if !st.openedItem {
			if st.itemID == "" {
				if mid := root.Get("message.id").String(); mid != "" {
					st.itemID = mid
				} else {
					st.itemID = "msg_" + st.responseID
				}
			}
			out = append(out, emitOutputItemAdded(st, nextSeq))
			out = append(out, emitContentPartAdded(st, nextSeq))
			st.openedItem = true
		}
		// Emit one delta per text content block. Drop tool_use / thinking
		// blocks — those stay inside CC.
		root.Get("message.content").ForEach(func(_, part gjson.Result) bool {
			if part.Get("type").String() != "text" {
				return true
			}
			text := part.Get("text").String()
			if text == "" {
				return true
			}
			st.textBuf.WriteString(text)
			delta := []byte(`{"type":"response.output_text.delta","sequence_number":0,"item_id":"","output_index":0,"content_index":0,"delta":"","logprobs":[]}`)
			delta, _ = sjson.SetBytes(delta, "sequence_number", nextSeq())
			delta, _ = sjson.SetBytes(delta, "item_id", st.itemID)
			delta, _ = sjson.SetBytes(delta, "delta", text)
			out = append(out, emit("response.output_text.delta", delta))
			return true
		})

	case "result":
		// Terminal frame. Close out the message item if one was opened, then
		// emit response.completed with the aggregated text and usage.
		if !st.closedItem && st.openedItem {
			out = append(out, emitOutputTextDone(st, nextSeq))
			out = append(out, emitContentPartDone(st, nextSeq))
			out = append(out, emitOutputItemDone(st, nextSeq))
			st.closedItem = true
		}
		// CC's `result.result` carries the canonical final string. Prefer it
		// over the accumulated textBuf when present, because tool-using runs
		// produce assistant deltas that aren't part of the final answer.
		final := root.Get("result").String()
		if final == "" {
			final = st.textBuf.String()
		}
		usage := root.Get("usage")
		out = append(out, emitCompleted(st, nextSeq, final, usage))
		st.completed = true
	}

	return out
}

// ConvertClaudeCLIResponseToOpenAIResponsesNonStream is the non-streaming
// counterpart. The ClaudeCLIExecutor synthesises an aggregated Responses-
// shaped payload itself in Execute, so this transform is a passthrough that
// hands the bytes back unchanged. Kept as an explicit function (rather than
// an identity registration) so the indirection is searchable.
func ConvertClaudeCLIResponseToOpenAIResponsesNonStream(_ context.Context, _ string, _, _, rawJSON []byte, _ *any) []byte {
	return rawJSON
}

// --- emit helpers ---------------------------------------------------------

func emitCreated(st *claudeCLIResponseState, nextSeq func() int) []byte {
	envelope := []byte(`{"type":"response.created","sequence_number":0,"response":{"id":"","object":"response","created_at":0,"status":"in_progress","background":false,"error":null,"output":[]}}`)
	envelope, _ = sjson.SetBytes(envelope, "sequence_number", nextSeq())
	envelope, _ = sjson.SetBytes(envelope, "response.id", st.responseID)
	envelope, _ = sjson.SetBytes(envelope, "response.created_at", st.createdAt)
	return emit("response.created", envelope)
}

func emitInProgress(st *claudeCLIResponseState, nextSeq func() int) []byte {
	envelope := []byte(`{"type":"response.in_progress","sequence_number":0,"response":{"id":"","object":"response","created_at":0,"status":"in_progress"}}`)
	envelope, _ = sjson.SetBytes(envelope, "sequence_number", nextSeq())
	envelope, _ = sjson.SetBytes(envelope, "response.id", st.responseID)
	envelope, _ = sjson.SetBytes(envelope, "response.created_at", st.createdAt)
	return emit("response.in_progress", envelope)
}

func emitOutputItemAdded(st *claudeCLIResponseState, nextSeq func() int) []byte {
	item := []byte(`{"type":"response.output_item.added","sequence_number":0,"output_index":0,"item":{"id":"","type":"message","status":"in_progress","content":[],"role":"assistant"}}`)
	item, _ = sjson.SetBytes(item, "sequence_number", nextSeq())
	item, _ = sjson.SetBytes(item, "item.id", st.itemID)
	return emit("response.output_item.added", item)
}

func emitContentPartAdded(st *claudeCLIResponseState, nextSeq func() int) []byte {
	part := []byte(`{"type":"response.content_part.added","sequence_number":0,"item_id":"","output_index":0,"content_index":0,"part":{"type":"output_text","annotations":[],"logprobs":[],"text":""}}`)
	part, _ = sjson.SetBytes(part, "sequence_number", nextSeq())
	part, _ = sjson.SetBytes(part, "item_id", st.itemID)
	return emit("response.content_part.added", part)
}

func emitOutputTextDone(st *claudeCLIResponseState, nextSeq func() int) []byte {
	done := []byte(`{"type":"response.output_text.done","sequence_number":0,"item_id":"","output_index":0,"content_index":0,"text":"","logprobs":[]}`)
	done, _ = sjson.SetBytes(done, "sequence_number", nextSeq())
	done, _ = sjson.SetBytes(done, "item_id", st.itemID)
	done, _ = sjson.SetBytes(done, "text", st.textBuf.String())
	return emit("response.output_text.done", done)
}

func emitContentPartDone(st *claudeCLIResponseState, nextSeq func() int) []byte {
	done := []byte(`{"type":"response.content_part.done","sequence_number":0,"item_id":"","output_index":0,"content_index":0,"part":{"type":"output_text","annotations":[],"logprobs":[],"text":""}}`)
	done, _ = sjson.SetBytes(done, "sequence_number", nextSeq())
	done, _ = sjson.SetBytes(done, "item_id", st.itemID)
	done, _ = sjson.SetBytes(done, "part.text", st.textBuf.String())
	return emit("response.content_part.done", done)
}

func emitOutputItemDone(st *claudeCLIResponseState, nextSeq func() int) []byte {
	final := []byte(`{"type":"response.output_item.done","sequence_number":0,"output_index":0,"item":{"id":"","type":"message","status":"completed","content":[{"type":"output_text","text":""}],"role":"assistant"}}`)
	final, _ = sjson.SetBytes(final, "sequence_number", nextSeq())
	final, _ = sjson.SetBytes(final, "item.id", st.itemID)
	final, _ = sjson.SetBytes(final, "item.content.0.text", st.textBuf.String())
	return emit("response.output_item.done", final)
}

func emitCompleted(st *claudeCLIResponseState, nextSeq func() int, finalText string, usage gjson.Result) []byte {
	envelope := []byte(`{"type":"response.completed","sequence_number":0,"response":{"id":"","object":"response","created_at":0,"status":"completed","background":false,"error":null,"output":[],"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`)
	envelope, _ = sjson.SetBytes(envelope, "sequence_number", nextSeq())
	envelope, _ = sjson.SetBytes(envelope, "response.id", st.responseID)
	envelope, _ = sjson.SetBytes(envelope, "response.created_at", st.createdAt)

	// Replay the assembled assistant message into the response.output array.
	if finalText != "" {
		item := []byte(`{"id":"","type":"message","status":"completed","content":[{"type":"output_text","annotations":[],"logprobs":[],"text":""}],"role":"assistant"}`)
		item, _ = sjson.SetBytes(item, "id", st.itemID)
		item, _ = sjson.SetBytes(item, "content.0.text", finalText)
		envelope, _ = sjson.SetRawBytes(envelope, "response.output.-1", item)
	}

	if usage.Exists() {
		in := usage.Get("input_tokens").Int()
		out := usage.Get("output_tokens").Int()
		envelope, _ = sjson.SetBytes(envelope, "response.usage.input_tokens", in)
		envelope, _ = sjson.SetBytes(envelope, "response.usage.output_tokens", out)
		envelope, _ = sjson.SetBytes(envelope, "response.usage.total_tokens", in+out)
	}
	return emit("response.completed", envelope)
}

// --- byte-fiddling helpers ------------------------------------------------

func startsWithDataTag(b []byte) bool {
	if len(b) < len(dataTag) {
		return false
	}
	for i, c := range dataTag {
		if b[i] != c {
			return false
		}
	}
	return true
}

func trimDataTag(b []byte) []byte {
	b = b[len(dataTag):]
	for len(b) > 0 && (b[0] == ' ' || b[0] == '\t') {
		b = b[1:]
	}
	for len(b) > 0 && (b[len(b)-1] == ' ' || b[len(b)-1] == '\t' || b[len(b)-1] == '\n' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	return b
}

// nowSuffix returns a stable-ish unique tail for synthetic ids when CC's
// init frame doesn't carry a session_id (older CC versions, edge cases).
func nowSuffix() string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	v := time.Now().UnixNano()
	out := make([]byte, 12)
	for i := range out {
		out[i] = charset[uint64(v)%uint64(len(charset))]
		v /= int64(len(charset))
	}
	return string(out)
}
