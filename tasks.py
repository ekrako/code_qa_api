from invoke import task


@task
def start(c):
    """Start the FastAPI server using uvicorn."""
    command = "uvicorn code_qa_api.main:app --host 0.0.0.0 --port 8000"
    print(f"Running: {command}")
    c.run(command, pty=True)


@task
def lint(c):
    """Run Ruff linter and MyPy type checker."""
    print("Running Ruff linter...")
    ruff_command = "ruff check src tests"
    print(f"Running: {ruff_command}")
    c.run(ruff_command, pty=True)

    print("\nRunning MyPy type checker...")
    mypy_command = "mypy src"
    print(f"Running: {mypy_command}")
    c.run(mypy_command, pty=True)


@task
def format(c):
    """Format code using Ruff formatter."""
    command = "ruff format src tests"
    print(f"Running: {command}")
    c.run(command, pty=True)


@task
def test(c):
    """Run tests using pytest."""
    command = "pytest -v --disable-warnings"
    print(f"Running: {command}")
    c.run(command, pty=True) 

@task
def evaluate(c):
    """Run evaluation script."""
    command = "python scripts/evaluate.py"
    print(f"Running: {command}")
    c.run(command, pty=True)