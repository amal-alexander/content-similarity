import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# App Title
st.title("📊 Content Similarity Checker (Cosine Similarity)")
st.write("Compare your content with a competitor's content to check similarity.")

# Input text areas
content1 = st.text_area("📝 Enter Your Content (Max 5000 words)", "", height=250, max_chars=25000)
content2 = st.text_area("📝 Enter Competitor's Content (Max 5000 words)", "", height=250, max_chars=25000)

# Compare button
if st.button("🔍 Compare Content"):
    if not content1.strip() or not content2.strip():
        st.warning("⚠️ Both content fields must be filled!")
    else:
        # Convert text into vector representation using TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([content1, content2])
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100  # Convert to percentage
        
        # Display result
        st.success(f"✅ Similarity Score: **{similarity_score:.2f}%**")

# Footer
st.write("---")
st.markdown("💡 Developed by **Amal Alexander** | Powered by **Streamlit & Scikit-Learn**")

# Instructions Section (Updated)
st.write("### 📌 How to Use This Tool")
st.markdown("""
1. **Enter Your Content** in the first text box.
2. **Enter Competitor's Content** in the second text box.
3. Click the **"Compare Content"** button to calculate similarity.
4. The tool will display a **Cosine Similarity Score (%)**.
""")
