from embedder import embed_image
from db_conn import get_conn_cursor

conn, cursor = get_conn_cursor()

def find_similar_images(query_image_path, k=3):
    query_embedding = embed_image(query_image_path)

    cursor.execute(
        """
        SELECT image_path, dataset, embedding <=> %s::vector AS distance
        FROM image_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (
            query_embedding.tolist(),
            query_embedding.tolist(),
            k
        )
    )

    return cursor.fetchall()

if __name__ == "__main__":
    query_image = "lol\\low\\22.png"
    results = find_similar_images(query_image, k=3)

    for i, (path, dataset, dist) in enumerate(results, 1):
        print(f"{i}. {path} | {dataset} | distance={dist:.4f}")

    cursor.close()
    conn.close()
