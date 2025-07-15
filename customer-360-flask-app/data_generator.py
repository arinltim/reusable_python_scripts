import json
import pandas as pd
import numpy as np

def generate_realistic_data(num_unique_customers=5):
    """
    Generates realistic but messy customer data from three distinct sources.
    """
    print("Generating synthetic raw data from 3 different sources...")

    base_profiles = [
        {'first': 'Jennifer', 'last': 'Mendoza', 'email_personal': 'jen.mendoza@gmail.com', 'email_work': 'j.mendoza@workplace.com', 'phone': '555-0101', 'city': 'New York'},
        {'first': 'Michael', 'last': 'Zhang', 'email_personal': 'mike.zhang@yahoo.com', 'email_work': 'michael.zhang@techcorp.io', 'phone': '555-0102', 'city': 'San Francisco'},
        {'first': 'Priya', 'last': 'Sharma', 'email_personal': 'priya.sharma@outlook.com', 'email_work': 'psharma@globalfoods.net', 'phone': '555-0103', 'city': 'Chicago'},
        {'first': 'David', 'last': 'Jones', 'email_personal': 'daveyjones@gmail.com', 'email_work': 'david.jones@finance.com', 'phone': '555-0104', 'city': 'Houston'},
        {'first': 'Samantha', 'last': 'Rodriguez', 'email_personal': 'sam.rodriguez@me.com', 'email_work': 'samantha.r@startup.co', 'phone': '555-0105', 'city': 'Los Angeles'}
    ]

    records = {
        'ecommerce': [],
        'mobile_app': [],
        'marketing': []
    }

    for i, profile in enumerate(base_profiles[:num_unique_customers]):
        # E-commerce Record (most complete)
        if np.random.rand() > 0.1: # 90% chance
            records['ecommerce'].append({
                'order_id': f'ord_10{i+1}',
                'customer_name': f"{profile['first']} {profile['last']}",
                'email': profile['email_personal'],
                'billing_address': f"{123+i} Main St, {profile['city']}",
                'phone_number': profile['phone'],
                'order_total': round(np.random.uniform(50, 500), 2)
            })

        # Mobile App Record (might use work email, has device ID)
        if np.random.rand() > 0.1: # 90% chance
            records['mobile_app'].append({
                'session_id': f'sess_abc_{i+1}',
                'user_id': f"user_{i+1}",
                'device_id': f"device_xyz_{i+1}",
                'email': profile['email_work'] if np.random.rand() > 0.5 else profile['email_personal'],
                'in_app_purchases': round(np.random.uniform(5, 99), 2),
                'time_spent_min': np.random.randint(5, 120)
            })

        # Marketing Record (often has incomplete names or just email)
        if np.random.rand() > 0.1: # 90% chance
            records['marketing'].append({
                'campaign_id': 'fall_promo_2025',
                'first_name': profile['first'] if np.random.rand() > 0.3 else '',
                'last_name': profile['last'],
                'email': profile['email_work'],
                'lead_source': 'Webinar',
                # **FIX:** Converted the numpy boolean to a standard Python bool
                'clicked_ad': bool(np.random.choice([True, False]))
            })

    # Add some noise/unmatched records
    records['marketing'].append({
        'campaign_id': 'winter_sale', 'first_name': 'Chris', 'last_name': 'Green',
        'email': 'chris.g@example.com', 'lead_source': 'Organic', 'clicked_ad': False
    })

    with open('raw_customer_data.json', 'w') as f:
        json.dump(records, f, indent=4)

    print("✓ Raw data saved to raw_customer_data.json")

if __name__ == '__main__':
    generate_realistic_data()