"""CLI script to generate embeddings and build the FAISS index."""

from retrieval.generate_embeddings import generate_embeddings
from retrieval.faiss_index import CommandIndex


def main():
    print("=== Step 1: Generate embeddings ===")
    embeddings, metadata = generate_embeddings()

    print("\n=== Step 2: Build FAISS index ===")
    index = CommandIndex.build(embeddings, metadata)
    index.save()

    print("\n=== Step 3: Verify ===")
    # Quick sanity check
    loaded = CommandIndex.load()
    results = loaded.search(embeddings[1], top_k=3)  # Search with first command's embedding
    first_cmd = loaded.id_to_name.get("1", "unknown")
    print(f"Self-search for '{first_cmd}':")
    for r in results:
        print(f"  {r['name']} (score: {r['score']:.4f})")

    print(f"\nIndex ready: {loaded.index.ntotal} vectors, {metadata['embedding_dim']}d")


if __name__ == "__main__":
    main()
