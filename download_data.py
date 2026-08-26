from src.data import ensure_dataset

movies_path, ratings_path = ensure_dataset()

print("MovieLens dataset is ready.")
print("Movies :", movies_path)
print("Ratings:", ratings_path)
