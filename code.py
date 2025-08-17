import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

df_books = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\MLproject\Books.csv")
df_ratings = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\MLproject\Ratings.csv")

df_books = df_books[['ISBN', 'Book-Title', 'Book-Author']]
df_books.dropna(inplace=True)

print("Books shape:", df_books.shape)
print("Ratings shape:", df_ratings.shape)

ratings_per_user = df_ratings['User-ID'].value_counts()
df_ratings_rm = df_ratings[~df_ratings['User-ID'].isin(ratings_per_user[ratings_per_user < 200].index)]

ratings_per_book = df_ratings['ISBN'].value_counts()
df_ratings_rm = df_ratings_rm[~df_ratings_rm['ISBN'].isin(ratings_per_book[ratings_per_book < 100].index)]

print("Filtered ratings shape:", df_ratings_rm.shape)

df = df_ratings_rm.pivot_table(index=['User-ID'], columns=['ISBN'], values='Book-Rating').fillna(0).T
df.index = df.join(df_books.set_index('ISBN'))['Book-Title']
df = df.sort_index()

print("Final pivot table shape:", df.shape)

model = NearestNeighbors(metric='cosine')
model.fit(df.values)

def get_recommends(title=""):
    if title not in df.index:
        print(f"⚠️ The book '{title}' does not exist in the dataset.")
        return None
    book = df.loc[title]
    distance, indice = model.kneighbors([book.values], n_neighbors=6)
    recommended_books = pd.DataFrame({
        'title': df.iloc[indice[0]].index.values,
        'distance': distance[0]
    }).sort_values(by='distance', ascending=False).head(5).values
    return [title, recommended_books]

def calculate_accuracy(input_title, recommended_books):
    if recommended_books is None:
        return 0
    _, book_array = recommended_books
    total_similarity = sum([1 - float(distance) for _, distance in book_array])
    return total_similarity / len(book_array)

def calculate_f_measure(relevant_books, recommended_books):
    true_positives = sum(1 for book in recommended_books if book in relevant_books)
    precision = true_positives / len(recommended_books) if recommended_books else 0
    recall = true_positives / len(relevant_books) if relevant_books else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f_measure

input_title = "Where the Heart Is (Oprah's Book Club (Paperback))"
recommended_books = get_recommends(input_title)

if recommended_books:
    print("\nRecommendations for:", input_title)
    print(recommended_books)
    accuracy = calculate_accuracy(input_title, recommended_books)
    print("Accuracy:", accuracy)

relevant_books = ["Where the Heart Is (Oprah's Book Club (Paperback))", "I'll Be Seeing You"]
recommended_books_list = ["Where the Heart Is (Oprah's Book Club (Paperback))", "The Surgeon", "Icy Sparks"]

f_measure = calculate_f_measure(relevant_books, recommended_books_list)
print("F-measure:", f_measure)
