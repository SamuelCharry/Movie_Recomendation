from rest_framework.response import Response
from rest_framework.decorators import api_view
import pandas as pd
from .config import movie_dict, similarity

movies = pd.DataFrame(movie_dict)

def get_movie_by_id(movie_id):
    # Getting the movie row
    movie = movies[movies['movie_id'] == movie_id]

    if not movie.empty:
        # returning the movie object
        return movie.iloc[0]
    return None

@api_view(['GET'])
def recommended(request, movie_id):
    recommended_movie_list_index = []

    try:
        movie_id = int(movie_id)
    except ValueError:
        return Response({"error": "Invalid movie ID"}, status=400)

    movie = get_movie_by_id(movie_id)
    if movie is None:
        return Response({"error": f"Movie with ID {movie_id} not found."}, status=404)

    try:
        # getting movie index (not movie_id)
        movie_index = movies[movies['movie_id'] == movie_id].index[0]
        distances = similarity[movie_index]
        # getting movie list with their index
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]

        for i in movies_list:
            recommended_movie_list_index.append(movies.iloc[i[0]].movie_id)

        return Response({"recommended_movies": recommended_movie_list_index})

    except IndexError:
        return Response({"error": "Movie not found"}, status=400)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

