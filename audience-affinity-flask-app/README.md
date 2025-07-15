# Audience Intelligence & Activation Platform

This document outlines the concepts, features, and data architecture of the Audience Intelligence & Activation Platform.

---

## 1. Use Case: The Audience Affinity Explorer

### Concept
A visual, interactive tool for discovering and understanding niche audience segments and their content affinities.

### Core Functionality

* **Flask UI:** A highly visual interface, perhaps using D3.js or a similar library, to represent audience clusters and their relationships.
* **Data Foundation (Snowflake & Databricks):** Run clustering algorithms (e.g., k-means, DBSCAN) in Databricks on your anonymized first-party data stored in Snowflake. The output would be a set of defined audience segments with their key characteristics.
* **Interactive Exploration:** The UI would allow users to:
    * Click on an audience cluster to see its defining attributes (demographics, interests, media consumption habits).
    * Explore the "affinity" between different audience segments and types of content or products.
    * Identify "lookalike" audiences within your data.
* **Export for Activation:** Users could select a discovered audience segment and export the corresponding user list (or an identifier) for activation in a marketing campaign.

### Unique Proposition
This tool democratizes data science. It provides a "no-code" way for marketing and content teams to perform sophisticated audience analysis, fostering a more data-driven approach to content creation and targeting. It directly empowers the "planning" phase of campaign management.

---

## 2. Key Metrics & Concepts

### Understanding the Affinity Score
In simple terms, the **Affinity Score** is an index that tells you how much more or less likely a specific audience segment is to engage with a particular media channel compared to your average customer.

The calculation is a standard industry formula:

> Affinity Score = (Segment’s Average Engagement on Channel / Overall Average Engagement on Channel) * 100

#### How to Interpret the Score:
* A score of **100** is the baseline. It means the segment's behavior on that channel is perfectly average.
* A score **above 100** indicates a strong preference or "affinity." For example, an affinity score of **160** for 'Social Media' means that the users in this segment spend **60% more time** on social media than the average customer.
* A score **below 100** indicates they engage *less* than average. A score of **75** means they spend **25% less time** on that channel compared to the average customer.

### How This Information Can Be Monetized (Business Value)
The Affinity Score is a powerful tool for media planning and maximizing your return on investment (ROI). It allows you to make data-driven decisions instead of guessing where to spend your advertising budget.

* **High Affinity Scores (e.g., > 120):** These are your "sweet spot" channels.
    * **Action:** Concentrate your ad spend here to reach this specific segment. Your advertising dollars are most effective on these channels because you are reaching a higher concentration of your target audience for less money.
    * **Example:** If "High-Value Buyers" have an affinity of 180 for Podcasts, you know that a podcast advertising campaign is an extremely efficient way to attract more high-value customers.

* **Low Affinity Scores (e.g., < 80):** These are channels to de-prioritize or avoid.
    * **Action:** Reduce or eliminate your ad spend on these channels when trying to target this specific segment.
    * **Example:** If the same "High-Value Buyers" have an affinity of 50 for Linear TV, you know that spending your budget on television ads to reach them would be highly inefficient and likely result in a low ROI.

In short, the Affinity Score helps you stop wasting money on ineffective channels and double down on the ones that are most likely to deliver results for each unique audience segment you discover.

---

## 3. Data Dictionary & Collection Methods

This section describes the columns in the `rich_audience_data.csv` file and how such data is typically collected.

### 3.1. First-Party (1st Party) Data

*Data collected directly from your own customers and systems.*

#### User Identification & Demographics
* `user_id`
    * **What it means:** A unique number that anonymously identifies a single user in your system.
    * **How it's collected:** This is 1st Party Data, typically generated and stored in your core user authentication or CRM system (e.g., Okta, Salesforce, or a custom user database) when a person creates an account.
* `age_group` & `income_bracket`
    * **What it means:** Standard demographic buckets that categorize users.
    * **How it's collected:** This can be a mix of 1st Party Data (provided by the user during sign-up) and 3rd Party Data (purchased from data providers and appended to user profiles).

#### Behavioral Engagement Metrics
* `time_on_social_media_min`, `time_on_streaming_video_min`, `time_on_podcasts_min`
    * **What they mean:** Measures of a user's content consumption habits (in minutes) over a specific period.
    * **How they're collected:** This is typically 1st Party Data if your company is the media platform (like YouTube or Spotify). It can also be acquired as 2nd/3rd party data from partners or data aggregators.
* `monthly_conversions`
    * **What it means:** A key performance indicator representing the number of valuable actions a user took in the last month (e.g., a purchase, a subscription).
    * **How it's collected:** This is almost always 1st Party Data tracked by your internal systems (e.g., e-commerce platform, app database).

#### Financial & Loyalty Metrics
* `customer_since_date` & `avg_monthly_spend`
    * **What they mean:** The date the user joined and their average monthly revenue contribution.
    * **How they're collected:** This is core 1st Party Data that comes directly from your billing or payment processing system (e.g., Stripe).
* `last_seen_days_ago`
    * **What it means:** A recency metric showing the days passed since the user's last active session. A strong predictor of churn.
    * **How it's collected:** This is 1st Party Data from your application's server logs or a session database.
* `support_tickets_raised`
    * **What it means:** The number of times a user has contacted customer support. Can indicate high engagement or high friction.
    * **How it's collected:** This is 1st Party Data pulled from your customer support platform (e.g., Zendesk).

#### The Target Variable
* `churned_in_last_90_days`
    * **What it means:** A historical label (`True`/`False`) that is the "answer" your churn prediction model learns from.
    * **How it's collected:** This is **Derived Data**. It is calculated by applying a business rule to historical data (e.g., `IF last_seen_days_ago > 90 THEN churned = True`).

### 3.2. Second-Party (2nd Party) Data

*Another company's first-party data that you acquire through a direct partnership.*

* `partner_purchase_categories` (e.g., 'electronics', 'apparel')
    * **What it means:** The types of products a user has purchased from a retail partner, providing insight into their broader spending habits.
    * **How it's collected:** A data-sharing agreement with an e-commerce or retail partner.
* `traveler_type` (e.g., 'business', 'leisure')
    * **What it means:** Categorizes the user's travel style, indicating disposable income and lifestyle.
    * **How it's collected:** A partnership with an airline, hotel chain, or online travel agency.
* `event_attendance_genres` (e.g., 'live music', 'sports')
    * **What it means:** The types of live events a user has purchased tickets for, revealing specific hobbies.
    * **How it's collected:** A partnership with a ticketing platform.

### 3.3. Third-Party (3rd Party) Data

*Data aggregated from many sources by a data provider, used to add broad demographic and psychographic context.*

* `household_size` & `presence_of_children`
    * **What they mean:** The number of people and a flag for children in the user's household, crucial for life-stage marketing.
    * **How it's collected:** Sourced from public records and consumer data panels (e.g., Acxiom, Experian).
* `education_level` & `marital_status`
    * **What they mean:** The user's highest level of education and marital status.
    * **How they're collected:** Sourced from public records and large-scale consumer surveys.
* `homeowner_status` & `estimated_net_worth`
    * **What they mean:** Whether a user owns or rents their home and their estimated financial standing.
    * **How they're collected:** Sourced from property records and financial data aggregators.
* `offline_purchase_intent` (e.g., 'in-market_for_car')
    * **What it means:** A flag indicating the user is actively researching a major purchase.
    * **How it's collected:** Built by data aggregators tracking online Browse behavior across thousands of websites.