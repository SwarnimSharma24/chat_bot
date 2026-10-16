import re


def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split text into chunks without breaking words.

    Args:
        text: The original text to split.
        chunk_size: Maximum number of characters in each chunk.
        overlap: Maximum number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """

    # Step 1: Validate the inputs.
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    # Step 2: Normalize whitespace.
    # Multiple spaces, tabs, and newlines become a single space.
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Step 3: Split the text into complete words.
    words = text.split()

    chunks = []
    start = 0

    # Step 4: Build chunks until all words are processed.
    while start < len(words):
        end = start

        # Expand the current chunk while it fits the character limit.
        while end < len(words):
            candidate = " ".join(words[start:end + 1])

            if len(candidate) > chunk_size:
                break

            end += 1

        # A single word cannot fit within the configured limit.
        if end == start:
            raise ValueError(
                f"Word exceeds chunk_size: {words[start]!r}"
            )

        # Step 5: Save the current chunk.
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Stop when we have processed every word.
        if end == len(words):
            break

        # Step 6: Find complete trailing words to overlap
        # with the next chunk.
        next_start = end

        while next_start > start:
            overlap_text = " ".join(words[next_start - 1:end])

            if len(overlap_text) > overlap:
                break

            next_start -= 1

        # Ensure the loop always makes progress.
        if next_start <= start:
            next_start = end

        start = next_start

    # Step 7: Validate the generated chunks.
    for chunk in chunks:
        assert len(chunk) <= chunk_size
        assert chunk == chunk.strip()
        assert not re.search(r"\s{2,}", chunk)

    return chunks


if __name__ == "__main__":
    sample_paragraph = """
Python functions are reusable blocks of code defined using the def keyword.
Functions can accept parameters and return results using the return keyword.
Python loops repeat operations. A for loop iterates over a sequence, while
a while loop repeats as long as its condition remains true.
Python lists are ordered, mutable collections that can store multiple values.
List indexing starts at zero.
"""

    # Repeat the paragraph with an explicit separator.
    sample_text = "\n".join([sample_paragraph] * 5)

    # Verify the source text before chunking.
    assert "return keyword" in sample_text
    assert "can accept" in sample_text
    assert "its condition" in sample_text
    assert "returnkeyword" not in sample_text
    assert "canaccept" not in sample_text
    assert "true.Python" not in sample_text

    print("Source text validation passed!")

    chunks = chunk_text(
        sample_text,
        chunk_size=300,
        overlap=50
    )

    # Verify every chunk and detect joined words.
    combined_text = " ".join(chunks)

    assert len(chunks) > 0
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert all(chunk == chunk.strip() for chunk in chunks)

    for bad_phrase in [
        "returnkeyword",
        "canaccept",
        "itscondition",
        "true.Python",
        "ofcode",
        "loopiterates",
        "mutablecollections",
    ]:
        assert bad_phrase not in combined_text, (
            f"Joined words detected: {bad_phrase!r}"
        )

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ({len(chunk)} characters) ---")
        print(chunk)

    print("\nAll chunk validation checks passed!")
