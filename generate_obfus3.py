import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11435/api/chat"
MODEL = "qwen3.5:9b"

INPUT_JSON = "600_human_text.json"
OUTPUT_JSON = "qwen_generated_obfus3.json"

TEXT_FIELD = "text"
ID_FIELD = "id"

START_INDEX = 350
END_INDEX = 376

LABEL_VALUE = 3

OVERWRITE_OUTPUT = False

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 10
NUM_PREDICT = 1200
MAX_INPUT_CHARS = 6000

def ask_qwen(prompt: str) -> str:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": "/no_think\n" + prompt,
                        }
                    ],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.4,
                        "num_ctx": 4096,
                        "num_predict": NUM_PREDICT,
                    },
                },
                timeout=3600,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:1000]}"
                )

            data = response.json()
            message = data.get("message", {})

            content = str(message.get("content", "")).strip()
            thinking = str(message.get("thinking", "")).strip()

            output = content

            if not output and thinking:
                output = thinking

            if not output:
                raise RuntimeError(f"Empty response from model. Full response: {data}")

            return output

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"All retries failed. Last error: {last_error}")


def get_genre_from_id(original_id) -> str:
    """
    Genre based on numeric id:
    1-1500     -> fiction
    1501-3000  -> news
    >3000      -> essay
    """
    try:
        numeric_id = int(original_id)
    except (ValueError, TypeError):
        return "unknown"

    if 1 <= numeric_id <= 1500:
        return "fiction"
    elif 1501 <= numeric_id <= 3000:
        return "news"
    elif numeric_id > 3000:
        return "essay"
    else:
        return "unknown"


def make_summary_prompt(text: str) -> str:
    text = str(text)[:MAX_INPUT_CHARS]

    return f"""
Summarize the text below into bullet points.

Include:
- main idea
- important events or arguments
- emotional tone
- writing style
- point of view
- tense
- any important details needed to rewrite the text later

Rules:
- Do not copy full sentences.
- Do not add new information.
- Output only bullet points.

Text:
{text}
""".strip()


def make_generation_prompt(summary: str) -> str:
    return f"""
Write a text from these bullet points, as a 10-year-old would write it
for a school assignment. The length has to be 500- 700 words. 
A 10-year-old writes like this:

- Concrete over abstract. They describe what happened, what someone did,
  what something looked like. Not what it meant or represented.
- Vocabulary is plain but not artificially tiny. They know words like
  "actually," "probably," "annoying," "weird." They don't know
  "perspective," "significant," "ultimately."
- Sentence length varies. Some are long and run on with "and" and "but"
  joining clauses that should be separate. Some are very short.
- Repetition is normal. The same word shows up four times in a paragraph
  because they didn't think to vary it.
- Opinions are stated flatly, without hedging. "It was boring." "He's
  mean." Not "I found it somewhat tedious."
- They get sidetracked. A detail reminds them of something else and
  they follow it for a sentence before coming back, or not coming back.
- Emotional range is real but narrow in expression: good, bad, fun,
  boring, scary, annoying, cool. Not "overwhelming," not "bittersweet."
- No thesis. No conclusion. They start with whatever comes first and
  stop when they run out.

Don't perform childishness. Just write the way an actual fifth-grader writes a paragraph for class.

Output only the text.

Bullet points:
{summary}
""".strip()


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    return data


def load_existing_output(path: str):
    output_path = Path(path)

    if not output_path.exists():
        return []

    with open(output_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            return []


def save_json(path: str, data: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def choose_items(items: list):
    if END_INDEX is None:
        return items[START_INDEX:]

    return items[START_INDEX:END_INDEX]


def main():
    input_path = Path(INPUT_JSON)
    output_path = Path(OUTPUT_JSON)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")

    if OVERWRITE_OUTPUT and output_path.exists():
        output_path.unlink()
        print(f"Deleted old output file: {OUTPUT_JSON}")

    input_items = read_json(INPUT_JSON)
    selected_items = choose_items(input_items)

    results = load_existing_output(OUTPUT_JSON)

    already_done_ids = {
        str(item.get("original_id"))
        for item in results
        if item.get("original_id") is not None
        and str(item.get("text", "")).strip()
    }

    print(f"Loaded {len(input_items)} input items.")
    print(f"Processing {len(selected_items)} selected items.")
    print(f"Input: {INPUT_JSON}")
    print(f"Output: {OUTPUT_JSON}")
    print(f"Model: {MODEL}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"num_predict: {NUM_PREDICT}")
    print(f"max input chars: {MAX_INPUT_CHARS}")

    start_time = time.time()

    progress_bar = tqdm(
        selected_items,
        total=len(selected_items),
        desc="Generating texts",
        unit="text",
    )

    for local_index, item in enumerate(progress_bar, start=START_INDEX):
        original_id = item.get(ID_FIELD, local_index)
        original_text = str(item.get(TEXT_FIELD, ""))
        genre = get_genre_from_id(original_id)

        progress_bar.set_postfix_str(f"id={original_id}, genre={genre}")

        if str(original_id) in already_done_ids:
            tqdm.write(f"Skipping id={original_id}: already processed")
            continue

        if not original_text.strip():
            tqdm.write(f"Skipping id={original_id}: empty text")
            continue

        tqdm.write(
            f"Processing id={original_id}, genre={genre}, input chars={len(original_text)}"
        )

        item_start_time = time.time()

        summary = ""
        generated_text = ""
        error_message = ""

        try:
            tqdm.write(f"Starting summary for id={original_id}")
            summary = ask_qwen(make_summary_prompt(original_text))
            tqdm.write(f"Finished summary for id={original_id}")

        except Exception as e:
            error_message = f"Summary error: {e}"
            tqdm.write(f"Error during summary for id={original_id}: {e}")

        if summary:
            try:
                tqdm.write(f"Starting generation for id={original_id}")
                generated_text = ask_qwen(make_generation_prompt(summary))
                tqdm.write(f"Finished generation for id={original_id}")

            except Exception as e:
                error_message = f"Generation error: {e}"
                tqdm.write(f"Error during generation for id={original_id}: {e}")

        result = {
            "original_id": original_id,
            "original_text": original_text,
            "summary": summary,
            "text": generated_text,
            "model": MODEL,
            "label": LABEL_VALUE,
            "genre": genre,
        }

        if error_message:
            result["error"] = error_message

        results.append(result)
        save_json(OUTPUT_JSON, results)

        if generated_text:
            already_done_ids.add(str(original_id))

        item_elapsed = time.time() - item_start_time
        progress_bar.set_postfix_str(
            f"id={original_id}, genre={genre}, last={item_elapsed:.1f}s"
        )

        time.sleep(1)

    total_elapsed = time.time() - start_time

    print("\nFinished.")
    print(f"Total time: {total_elapsed / 60:.1f} minutes")

    if selected_items:
        print(f"Average time per selected text: {total_elapsed / len(selected_items):.1f} seconds")

    print(f"Output saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()