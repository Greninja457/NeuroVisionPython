import os
from embedder import embed_image
from db_conn import get_conn_cursor

conn, cursor = get_conn_cursor()

def insert_embedding(dataset, image_path, embedding):
    cursor.execute(
        """
        INSERT INTO image_embeddings (dataset, image_path, embedding)
        VALUES (%s, %s, %s)
        """,
        (
            dataset,
            image_path,
            embedding.tolist()
        )
    )

def add_images_from_folder(folder_path, source_name):
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, file)

            embedding = embed_image(path)

            insert_embedding(source_name,path, embedding)
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # add_images_from_folder("coco", "coco")
    add_images_from_folder("lol\\high", "lol_gt")
    # add_images_from_folder("pro", "pro")