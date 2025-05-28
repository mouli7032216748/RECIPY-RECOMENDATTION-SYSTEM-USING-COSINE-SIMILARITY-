import pandas as pd
import re
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset with correct encoding
df = pd.read_csv("recipes.csv", encoding="latin1")

# Fill missing values
df["Ingredients"] = df["Ingredients"].fillna("")
df["Instructions"] = df["Instructions"].fillna("No instructions available.")
df["Difficulty"] = df["Difficulty"].fillna("Unknown")
df["Quantity per Serving"] = df["Quantity per Serving"].fillna("")

# Preprocess ingredients to lowercase
df["Ingredients"] = df["Ingredients"].apply(lambda x: x.lower() if isinstance(x, str) else x)

# Vectorize ingredients
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = vectorizer.fit_transform(df["Ingredients"])

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input and normalize
        user_ingredients_input = request.form["ingredients"].lower()
        user_ingredients = [i.strip() for i in re.split(r",\s*|\n", user_ingredients_input)]
        servings = int(request.form["servings"])

        # Transform user ingredients into vector
        user_vector = vectorizer.transform([" ".join(user_ingredients)])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        # Filter top matches (with optional threshold)
        top_indices = similarity_scores.argsort()[::-1][:5]

        suggestions = df.iloc[top_indices].copy()

        # Helper to adjust quantities
        def adjust_quantities(recipe_row, servings):
            ingredients_with_qty = recipe_row["Quantity per Serving"].split(", ")
            adjusted_ingredients = []

            for item in ingredients_with_qty:
                try:
                    ingredient, qty = item.rsplit("-", 1)
                    qty_value, unit = parse_quantity(qty)
                    adjusted_qty = qty_value * servings
                    adjusted_ingredients.append(f"{ingredient.strip()} - {adjusted_qty:.2f}{unit}")
                except ValueError:
                    adjusted_ingredients.append(item)

            return ", ".join(adjusted_ingredients)

        # Quantity parser (handles mixed units)
        def parse_quantity(qty_str):
            match = re.match(r"([\d.]+)\s*([a-zA-Z]*)", qty_str.strip())
            if match:
                return float(match.group(1)), match.group(2)
            return 0, ""

        # Apply adjustments
        suggestions["Adjusted Ingredients"] = suggestions.apply(
            lambda row: adjust_quantities(row, servings), axis=1
        )

        recipe_results = suggestions[[
            "Recipe Name", "Ingredients", "Quantity per Serving",
            "Instructions", "Time (mins)", "Difficulty", "Adjusted Ingredients"
        ]]

        return render_template("results.html", recipes=recipe_results.to_dict(orient="records"), servings=servings)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
