import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker to generate mock data
fake = Faker()

# Define the number of mock users
num_users = 2000

# --- Define richer categories and data points ---
age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
income_brackets = ['Low', 'Medium', 'High']

# --- NEW: Data pools for 2nd and 3rd party data ---
# 2nd Party Data Pools
partner_purchase_pool = ['Electronics', 'Apparel', 'Home Goods', 'Groceries', 'Books', 'Fitness']
traveler_type_pool = ['Business', 'Leisure', 'Family', 'Solo', 'Backpacker']

# 3rd Party Data Pools
education_pool = ['High School', 'Bachelors Degree', 'Masters Degree', 'PhD']
homeowner_pool = ['Own', 'Rent']
purchase_intent_pool = ['None', 'In-Market for Car', 'In-Market for Home', 'Planning Vacation', 'Seeking Insurance']
# ----------------------------------------------------

print(f"Generating enriched mock data for {num_users} users...")

# Generate the data
data = []
for i in range(num_users):
    # Core Demographics
    age = random.choice(age_groups)
    income = random.choice(income_brackets)

    # Engagement Data (correlated with age/income for realism)
    time_on_social = random.randint(10, 240) + (20 if age in ['18-24', '25-34'] else 0)
    time_on_video = random.randint(5, 300) + (30 if income in ['High'] else 0)
    time_on_podcast = random.randint(0, 180) + (40 if age in ['35-44', '45-54'] else 0)

    # Monetization Data
    customer_since = fake.date_time_between(start_date='-4y', end_date='-3m')
    avg_monthly_spend = max(10, random.gauss(50, 20) + (50 if income == 'High' else 0) + (20 if age == '35-44' else -10))

    # Churn Predictor Features
    last_seen_days_ago = random.randint(1, 120)
    support_tickets_raised = random.choices([0, 1, 2, 3, 4], weights=[0.6, 0.2, 0.1, 0.05, 0.05], k=1)[0]

    # Target Variable for Churn Model (heuristic)
    churned = (last_seen_days_ago > 90 and support_tickets_raised > 1) or (last_seen_days_ago > 100)

    # --- NEW: Generation of 2nd and 3rd party data points ---
    # 2nd Party Data
    partner_purchases = random.sample(partner_purchase_pool, k=random.randint(1, 2))
    traveler_type = random.choice(traveler_type_pool) if random.random() > 0.4 else 'Unknown' # Simulates sparse data

    # 3rd Party Data
    household_size = random.randint(1, 5)
    education = random.choice(education_pool)
    homeowner = random.choice(homeowner_pool)
    purchase_intent = random.choices(purchase_intent_pool, weights=[0.65, 0.1, 0.1, 0.1, 0.05], k=1)[0]
    # -----------------------------------------------------------

    data.append({
        # Original 1st Party Data
        'user_id': 1000 + i,
        'age_group': age,
        'income_bracket': income,
        'time_on_social_media_min': time_on_social,
        'time_on_streaming_video_min': time_on_video,
        'time_on_podcasts_min': time_on_podcast,
        'monthly_conversions': max(0, int(random.gauss(5, 3))),
        'customer_since_date': customer_since,
        'avg_monthly_spend': round(avg_monthly_spend, 2),
        'last_seen_days_ago': last_seen_days_ago,
        'support_tickets_raised': support_tickets_raised,
        'churned_in_last_90_days': churned,

        # --- NEW: Added columns ---
        'partner_purchase_categories': ", ".join(partner_purchases),
        'traveler_type': traveler_type,
        'household_size': household_size,
        'education_level': education,
        'homeowner_status': homeowner,
        'offline_purchase_intent': purchase_intent
    })

df = pd.DataFrame(data)

# Save to a new CSV file
df.to_csv('rich_audience_data.csv', index=False)
print("SUCCESS: Enriched data generated and saved to 'rich_audience_data.csv'")