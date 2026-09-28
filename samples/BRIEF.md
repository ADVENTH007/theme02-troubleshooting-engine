# Smart Guided Troubleshooting Engine — topic pack

Extracted from `Theme02_Input_Kit.zip` (nested inside participant-kit-all-themes).
The parent kit's harness/agent/scenarios/docs belong to a different theme
(Theme 5: Interruptible Agents) and are left out.

## What is in here
| file | purpose |
|---|---|
| `input.txt` | 20 raw user queries (screen problems on Galaxy phones/tablets) |
| `siis_responses.json` | 20 pre-fetched knowledge-store (SIIS) responses, one per query: `id`, `original_query`, `siis_response{title, content}`. This is the payload `POST /v1/troubleshoot` must accept |
| `deeplinks.json` | Catalog of 578 masked Galaxy Settings deeplinks (act + validation). Match on `description`, `message`, `qna_description`, `originalType`; copy URIs verbatim. `bixby://dummy_positive` is the only generic placeholder |
| `schema.py` | Pydantic response schema (`ContextDeeplinkResponse` -> `Goal` -> `Action` -> `StepGroup`) — validate every response against it |
| `sample_output.json` | One worked example of the expected output |

## Task, as far as the files show
Turn a user query + raw SIIS article text into a structured guided flow:
1. `contexts[]` = list of `Goal` (goal, title, score 0-1)
2. each Goal has `actions[]` (actionName, description, category: auto / manual / critical)
3. each Action has `stepGroups[]` = `steps[]` (plain-language instructions) plus optional
   `actionableDeeplink` (do it) and `validationDeeplink` (check it worked: key, resultType, condition, value)
4. Only use steps that are in the SIIS text — do not invent steps.
5. Use the deeplink catalog for actionable/validation links; no matching entry -> dummy placeholder
   with your own 5-7 word description/message.

## Things to note
- The kit contains no written problem statement or scoring doc for this theme, so the above is inferred
  from `schema.py`, `sample_output.json` and the `_readme` fields. Check the event brief for the rest
  (scoring, latency, whether "voice" input is in scope).
- Some SIIS matches look off-topic (e.g. row_1 query is a blank screen in Gmail, but the article
  returned is about email server errors) — handling weak retrieval is likely part of the challenge.
- Query 16 in `input.txt` bundles three complaints in one line.
