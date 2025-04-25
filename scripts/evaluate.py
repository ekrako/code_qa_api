import asyncio
import re  # Import re for pattern matching
import subprocess  # Import subprocess to run git clone
import sys
from pathlib import Path

import httpx
from sentence_transformers import SentenceTransformer, util

from code_qa_api.core.config import settings  # Load settings like API base URL and QA path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

QA_FILE_PATH = settings.qa_data_path
API_URL = f"{settings.api_base_url}/api/answer"

# Load a sentence transformer model for semantic similarity
# Using a smaller model for efficiency in evaluation
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define the repository URL
QA_REPO_URL = settings.qa_repo_url


def load_qa_data(dir_path: Path) -> list | None:
    """Loads QA pairs from .q.md and .a.md files in the specified directory. Clones the repo if not found."""
    if not dir_path.is_dir():
        print(f"QA directory not found at {dir_path}. Attempting to clone from {QA_REPO_URL}...")
        try:
            # Ensure parent directory exists
            dir_path.parent.mkdir(parents=True, exist_ok=True)
            # Clone the repository
            result = subprocess.run(
                ["git", "clone", QA_REPO_URL, str(dir_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"Successfully cloned QA repository to {dir_path}.")
            else:
                print(f"Warning: Git clone completed with non-zero exit code: {result.returncode}")
            # Check again if the directory exists after cloning
            if not dir_path.is_dir():
                 print(f"Error: Directory {dir_path} still not found after attempting clone.")
                 return None

        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            print(f"Stderr: {e.stderr}")
            print("Please ensure Git is installed and you have permissions to clone.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during cloning: {e}")
            return None


    qa_pairs = {}
    q_pattern = re.compile(r"(\d+)\.q\.md")
    a_pattern = re.compile(r"(\d+)\.a\.md")

    try:
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                q_match = q_pattern.match(file_path.name)
                a_match = a_pattern.match(file_path.name)

                if q_match:
                    index = q_match.group(1)
                    if index not in qa_pairs:
                        qa_pairs[index] = {}
                    with open(file_path, "r", encoding="utf-8") as f:
                        qa_pairs[index]["question"] = f.read().strip()
                elif a_match:
                    index = a_match.group(1)
                    if index not in qa_pairs:
                        qa_pairs[index] = {}
                    with open(file_path, "r", encoding="utf-8") as f:
                        qa_pairs[index]["answer"] = f.read().strip()

        # Convert to list and validate
        qa_data = []
        for index, pair in sorted(qa_pairs.items()):
            if "question" in pair and "answer" in pair:
                qa_data.append({"question": pair["question"], "answer": pair["answer"]})
            else:
                print(
                    f"Warning: Missing question or answer for index {index} in {dir_path}"
                )

        if not qa_data:
            print(f"Error: No valid QA pairs found in {dir_path}.")
            return None

        return qa_data

    except Exception as e:
        print(f"Error reading files from QA directory {dir_path}: {e}")
        return None


async def get_answer_from_api(question: str, client: httpx.AsyncClient) -> str | None:
    try:
        response = await client.post(
            API_URL, json={"question": question}, timeout=120.0
        )
        response.raise_for_status()  # Raise exception for 4xx or 5xx errors
        return response.json().get("answer")
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
    except httpx.HTTPStatusError as exc:
        print(
            f"Error response {exc.response.status_code} while"
            f" requesting {exc.request.url!r}: {exc.response.text}"
        )
    except Exception as e:
        print(f"An unexpected error occurred calling the API: {e}")
    return None


def calculate_similarity(answer: str, reference: str) -> float:
    emb1 = similarity_model.encode(answer, convert_to_tensor=True)
    emb2 = similarity_model.encode(reference, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb1, emb2)
    return cosine_scores.item()


async def run_evaluation():
    print("Starting evaluation...")
    qa_data = load_qa_data(QA_FILE_PATH)
    if not qa_data:
        return

    print(f"Loaded {len(qa_data)} question/answer pairs.")
    scores = []
    results = []

    async with httpx.AsyncClient() as client:
        for i, item in enumerate(qa_data):
            question = item["question"]
            reference_answer = item["answer"]
            print(f"\nProcessing question {i+1}/{len(qa_data)}: {question}")

            generated_answer = await get_answer_from_api(question, client)

            if generated_answer:
                similarity_score = calculate_similarity(
                    generated_answer, reference_answer
                )
                scores.append(similarity_score)
                print(
                    f"  Reference Answer: {reference_answer[:100]}..."
                    if len(reference_answer) > 100
                    else reference_answer
                )
                print(
                    f"  Generated Answer: {generated_answer[:100]}..."
                    if len(generated_answer) > 100
                    else generated_answer
                )
                print(f"  Similarity Score: {similarity_score:.4f}")
                results.append(
                    {
                        "question": question,
                        "reference": reference_answer,
                        "generated": generated_answer,
                        "score": similarity_score,
                    }
                )
            else:
                print("  Failed to get answer from API.")
                scores.append(0.0)  # Assign 0 score if API fails
                results.append(
                    {
                        "question": question,
                        "reference": reference_answer,
                        "generated": None,
                        "score": 0.0,
                    }
                )

    if scores:
        average_score = sum(scores) / len(scores)
        print("\nEvaluation Complete.")
        print(f"Average Semantic Similarity Score: {average_score:.4f}")
    else:
        print("\nEvaluation could not be completed (no scores calculated).")

    # Optional: Save detailed results to a file
    # with open("evaluation_results.json", "w") as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Check if API is likely running
    try:
        response = httpx.get(settings.api_base_url + "/docs", timeout=5.0)
        if response.status_code != 200:
            print(
                f"Warning: API server at {settings.api_base_url} might not be running or /docs is unavailable (Status: {response.status_code}). Proceeding anyway."
            )
    except httpx.RequestError as e:
        print(
            f"Error: Could not connect to API server at {settings.api_base_url}. Please ensure it's running before evaluation."
        )
        print(f"Details: {e}")
        sys.exit(1)

    asyncio.run(run_evaluation())
