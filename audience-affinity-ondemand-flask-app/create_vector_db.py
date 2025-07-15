import pandas as pd
from faker import Faker
import random
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from sklearn.decomposition import PCA

# --- 1. Generate Rich User Profiles (Unchanged) ---
print("--- Step 1: Generating Rich User Profiles ---")
# (The data generation code from the previous version is the same)
fake = Faker()
num_users = 5000
interests_pool = ['hiking', 'photography', 'cooking', 'live music', 'tech gadgets', 'wine tasting', 'yoga', 'DIY projects', 'classic films', 'sustainable living', 'fantasy novels']
activity_pool = ['bought a new camera', 'attended a marketing conference', 'booked a flight to Italy', 'researched sustainable investment funds', 'started a home renovation project', 'signed up for a marathon']
data = []
for i in range(num_users):
    name, job, city = fake.name(), fake.job(), fake.city()
    user_interests = random.sample(interests_pool, k=random.randint(2, 4))
    recent_activity = random.choice(activity_pool)
    profile_text = (f"{name} is a {job} living in {city}. They are interested in {', '.join(user_interests)}. Their recent activity includes: {recent_activity}.")
    data.append({'user_id': 1000 + i, 'name': name, 'job': job, 'city': city, 'interests': ", ".join(user_interests), 'profile_text': profile_text})
df = pd.DataFrame(data)
print(f"Successfully generated {num_users} user profiles.")

# --- 2. Create Vector Embeddings (Unchanged) ---
print("\n--- Step 2: Creating Vector Embeddings ---")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['profile_text'].tolist(), show_progress_bar=True)

# --- 3. NEW: Add PCA coordinates for visualization ---
print("\n--- Step 3: Generating PCA coordinates for visualization ---")
pca = PCA(n_components=2)
vis_dims = pca.fit_transform(embeddings)
df['pca1'] = vis_dims[:, 0]
df['pca2'] = vis_dims[:, 1]
print("PCA coordinates generated and added to user data.")

# --- 4. Build and Save FAISS Index and User Data ---
print("\n--- Step 4: Building and Saving FAISS Index & User Data ---")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings).astype('float32'))
faiss.write_index(index, 'users.faiss')
print(f"FAISS index with {index.ntotal} vectors saved to 'users.faiss'")
with open('users.pkl', 'wb') as f:
    pickle.dump(df, f)
print("User data (including PCA coordinates) saved to 'users.pkl'")

print("\nSetup Complete!")