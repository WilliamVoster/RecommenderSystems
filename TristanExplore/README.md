# Content-Based Filtering for News Recommendation

## Overview
This document describes the implementation of a **content-based filtering** approach for a news recommendation system. The dataset consists of:
- **News articles** with **keywords** and **genres**.
- **User behavior data**, tracking clicks on articles.

The goal is to recommend news articles based on a user's **past reading preferences** by analyzing the content of articles they clicked on.

---

## 1. Feature Representation of News Articles
To compare articles, we must convert their textual and categorical data into numerical vectors. We achieve this using **Bag of Words (BoW)**, combining **TF-IDF for keywords** and **one-hot encoding for genres**.

### **Bag of Words (BoW) Representations**

1. **Unweighted (Binary) BoW**
   - Checks only if a word exists (1) or not (0).
   - Ignores word frequency.
   
2. **Term Frequency (TF)**
   - Counts how often each word appears.
   - Higher frequency = Higher importance.

3. **TF-IDF (Term Frequency - Inverse Document Frequency)**
   - Weighs words based on their frequency in a document vs. their rarity across all documents.
   - Common words like "the" get lower weights.

### **Combining Features for News Articles**
Each article is represented as a **feature vector**, combining:
- **TF-IDF representation of keywords** (to capture article content)
- **One-hot encoding of genres** (to capture categorical information)

Example representation:
```
Article 1: [0.6, 0.7, 0.5] (TF-IDF keywords) + [1, 0, 0] (One-hot genres)
Article 2: [0.8, 0.9, 0.6] (TF-IDF keywords) + [0, 1, 0] (One-hot genres)
```

---

## 2. User Profile Construction
A **user profile** is built by aggregating the feature vectors of the articles they have clicked on.

### **User Profile Calculation**
For a user who clicked on articles \( A_1, A_2, ..., A_N \):

\[
User\_Profile = \frac{1}{N} \sum_{j \in Clicked\_Articles} Article\_Vector_j
\]

This creates a **weighted preference vector** representing the user’s interests.

---

## 3. Scoring Function: Cosine Similarity
To find the best news recommendations, we compute **cosine similarity** between the user's profile vector and each article vector.

### **Cosine Similarity Formula**
\[
s(x_i, x_j) = \frac{x_i \cdot x_j}{||x_i|| \cdot ||x_j||}
\]

where:
- **\( x_i \)** → User profile vector
- **\( x_j \)** → News article vector
- **Output** → A similarity score between **0 and 1**

### **Example Calculation**
If a user has profile vector **[0.5, 0, 0.5]**, and an article has vector **[1, 0, 0]**, the similarity score is:

\[
s(User, Article) = \frac{(0.5 \times 1) + (0 \times 0) + (0.5 \times 0)}{\sqrt{0.5^2 + 0^2 + 0.5^2} \times \sqrt{1^2 + 0^2 + 0^2}} = 0.707
\]

A higher score means the article is **more relevant** for the user.

---

## 4. Generating Recommendations
1. Compute **cosine similarity** between the user's profile and all articles.
2. **Rank articles** by similarity score.
3. Recommend the **top N most similar articles** to the user.

---

## 5. Summary of Content-Based Filtering Workflow
### **Step 1: Preprocessing News Articles**
- Extract **TF-IDF features** from keywords.
- Encode **genres using one-hot encoding**.
- Create **feature vectors** for all articles.

### **Step 2: Building User Profiles**
- Aggregate feature vectors of **clicked articles**.
- Compute an **average vector** to represent the user.

### **Step 3: Computing Similarities**
- Measure **cosine similarity** between **user profiles and articles**.
- Rank articles by similarity.

### **Step 4: Recommending News Articles**
- Recommend **top N most similar articles**.
- Update user profiles **as they interact with more news**.

---

## 6. Enhancements & Considerations
🚀 **Cold Start Problem?** → Use **popular articles** for new users.
🚀 **Diversity?** → Mix **high-score** and **exploratory** recommendations.
🚀 **Real-Time Updates?** → Update user profiles dynamically after each click.

By following this structured approach, we create a **scalable and adaptive content-based news recommender system**! 🚀
