# Outside-Diff Impact Slicing for AI Code Reviews

An AI-powered code review tool that finds real bugs by analyzing the boundaries between changed code and its callers/callees. Instead of just looking at the diff, this tool performs **Outside-Diff Impact Slicing** to catch contract mismatches, logic errors, and integration bugs that traditional diff-based reviews miss.

## What It Does

The tool analyzes git diffs and builds context by:
- Identifying changed functions/classes in your commits
- Finding **callees** (definitions your changed code calls)
- Finding **callers** (code that calls your changed functions)
- Sending structured context to GPT-5-mini to detect bugs at these boundaries

This catches bugs like:
- Functions called with wrong parameter counts after signature changes
- Callers not updated when function signatures change
- Contract violations between changed code and its dependencies

## Files

- `review_demo.py` - Main script implementing the Outside-Diff Impact Slicing technique
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+
- Git repository with at least one commit to diff against
- OpenAI API key (for GPT-5-mini access)

## Installation

```bash
# Clone the repository
git clone https://github.com/coderabbitai/odsc-west-2025.git
cd odsc-west-2025

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Note:** This tool currently analyzes Python codebases only.

Run the script from any Python git repository with uncommitted changes or recent commits:

```bash
python review_demo.py
```

When prompted, enter your OpenAI API key. The script will:
1. Extract changed lines from `git diff HEAD~1`
2. Build a call graph of your Python codebase
3. Identify impact files (callers and callees)
4. Generate structured context with the diff, changed code, and impact code
5. Send the context to GPT-5-mini for bug analysis
6. Output JSON findings with bug categories, summaries, and suggested fixes

### Output Format

The tool outputs JSON with structured bug reports:
```json
{
  "bugs": [
    {
      "changed_file": "path/to/file.py",
      "changed_lines": "73",
      "bug_category": "contract-mismatch",
      "summary": "Function called with wrong parameter types",
      "comment": "Detailed explanation with evidence...",
      "diff_fix_suggestion": "--- a/file.py\n+++ b/file.py\n..."
    }
  ]
}
```

## Limitations

- Currently supports Python codebases only
- Works best on focused PRs (10-50 changed lines)
- Large PRs may exceed context window limits

## Learn More

For a detailed explanation of the technique, see the accompanying article on context engineering for AI code reviews.

## Presented at ODSC West 2025

This demo accompanies the talk **"Context Engineering for AI Code Reviews with MCP, LLMs, and Open-Source DevOps Tooling"** at ODSC AI West.
