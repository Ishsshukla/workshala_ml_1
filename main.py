# main.py

from fastapi import FastAPI
from recommendation_module import create_similarity_matrix, search_courses
from data_loader import load_dataframe

app = FastAPI()

#  DataFrame
df = load_dataframe()

# similarity matrix
similarity_matrix = create_similarity_matrix(df)

@app.get("/recommend/{keyword}")
async def get_recommendations(keyword: str):
    try:
        matching_courses = search_courses(keyword, df)
        if not matching_courses:
            return {"message": f"No matching courses found for {keyword}"}

       

        recommended_courses = recommend_courses(keyword, similarity_matrix, df, top_n=5)
        return {"message": f"Recommendations for {keyword}", "courses": recommended_courses}
    
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
