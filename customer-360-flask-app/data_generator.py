# data_generator.py
import json
import random
import uuid

def generate_realistic_data(num_total_records=10000, num_unique_customers=27):
    """
    Generates data where device_id ONLY appears in the media_consumption source.
    """

    base_profiles = [
        {'first': 'Jennifer', 'last': 'Mendoza', 'aliases': ['Jen Mendoza', 'J. Mendoza'], 'emails': ['jen.mendoza@gmail.com', 'j.mendoza@workplace.com'], 'phone': '555-0101', 'device_ids': [f'device_jen_{i}' for i in range(3)]},
        {'first': 'Michael', 'last': 'Zhang', 'aliases': ['Mike Zhang', 'Michael Z.'], 'emails': ['mike.zhang@yahoo.com', 'michael.zhang@techcorp.io'], 'phone': '555-0102', 'device_ids': [f'device_mike_{i}' for i in range(3)]},
        {'first': 'Priya', 'last': 'Sharma', 'aliases': ['Priya S.', 'P. Sharma'], 'emails': ['priya.sharma@outlook.com', 'psharma@globalfoods.net'], 'phone': '555-0103', 'device_ids': [f'device_priya_{i}' for i in range(3)]},
        {'first': 'David', 'last': 'Jones', 'aliases': ['Davey Jones', 'David L. Jones'], 'emails': ['daveyjones@gmail.com', 'david.jones@finance.com'], 'phone': '555-0104', 'device_ids': [f'device_dave_{i}' for i in range(3)]},
        {'first': 'Samantha', 'last': 'Rodriguez', 'aliases': ['Sam Rodriguez', 'S. Rodriguez'], 'emails': ['sam.rodriguez@me.com', 'samantha.r@startup.co'], 'phone': '555-0105', 'device_ids': [f'device_sam_{i}' for i in range(3)]},
        {'first': 'Christopher', 'last': 'Lee', 'aliases': ['Chris Lee', 'C. Lee'], 'emails': ['chris.lee@icloud.com', 'christopher.lee@lawfirm.com'], 'phone': '555-0106', 'device_ids': [f'device_chris_{i}' for i in range(3)]},
        {'first': 'Aisha', 'last': 'Khan', 'aliases': ['Aisha K.', 'A. Khan'], 'emails': ['akhan@university.edu', 'aisha.khan@research.org'], 'phone': '555-0107', 'device_ids': [f'device_aisha_{i}' for i in range(3)]},
        {'first': 'Daniel', 'last': 'Kim', 'aliases': ['Dan Kim', 'Daniel K.'], 'emails': ['dan.kim@gmail.com', 'dkim@designs.com'], 'phone': '555-0108', 'device_ids': [f'device_dan_{i}' for i in range(3)]},
        {'first': 'Olivia', 'last': 'Chen', 'aliases': ['Liv Chen', 'O. Chen'], 'emails': ['olivia.chen@icloud.com', 'chen.o@pharma.com'], 'phone': '555-0109', 'device_ids': [f'device_olivia_{i}' for i in range(3)]},
        {'first': 'William', 'last': 'Taylor', 'aliases': ['Will Taylor', 'Bill Taylor'], 'emails': ['will.taylor@yahoo.com', 'william.t@consulting.com'], 'phone': '555-0110', 'device_ids': [f'device_will_{i}' for i in range(3)]},
        {'first': 'Sophia', 'last': 'Nguyen', 'aliases': ['Sophie Nguyen', 'S. Nguyen'], 'emails': ['sophia.n@gmail.com', 's.nguyen@fashion.co'], 'phone': '555-0111', 'device_ids': [f'device_sophia_{i}' for i in range(3)]},
        # ** Two distinct people named James Smith with their own unique device IDs
        {'first': 'James', 'last': 'Smith', 'aliases': ['Jim Smith'], 'emails': ['james.smith.1985@gmail.com', 'jsmith@construction.co'], 'phone': '555-0126', 'device_ids': ['device_tv_abc', 'device_mobile_123']},
        {'first': 'James', 'last': 'Smith', 'aliases': ['James S.', 'J. Allen Smith'], 'emails': ['j.a.smith@university.edu', 'james.allen.smith@proton.me'], 'phone': '555-0127', 'device_ids': ['device_smarttv_xyz', 'device_tablet_456']}
    ]

    active_profiles = base_profiles[:num_unique_customers]

    print(f"Generating {num_total_records} records that will resolve to {len(active_profiles)} unique customers...")

    records = {
        'ecommerce': [],
        'mobile_app': [],
        'marketing': [],
        'media_consumption': []
    }

    for _ in range(num_total_records):
        profile = random.choice(active_profiles)
        source = random.choice(['ecommerce', 'mobile_app', 'marketing', 'media_consumption'])

        new_record = {}

        if source == 'ecommerce':
            new_record = {
                'order_id': f'ord_{uuid.uuid4().hex[:8]}',
                'customer_name': random.choice([f"{profile['first']} {profile['last']}", random.choice(profile['aliases'])]),
                'email': random.choice(profile['emails']),
                'order_total': round(random.uniform(20, 800), 2)
            }
            if random.random() > 0.3: new_record['phone_number'] = profile['phone']

        elif source == 'mobile_app':
            new_record = {
                'session_id': f'sess_{uuid.uuid4().hex[:10]}',
                'email': random.choice(profile['emails']),
                'time_spent_min': random.randint(1, 180),
                # **FIX:** device_id is REMOVED from this source
            }

        elif source == 'marketing':
            new_record = {
                'campaign_id': random.choice(['spring_2025', 'q3_promo', 'holiday_push']),
                'email': random.choice(profile['emails']),
                'clicked_ad': random.choice([True, False])
            }
            if random.random() > 0.5:
                new_record['first_name'] = profile['first']
                new_record['last_name'] = profile['last']

        elif source == 'media_consumption':
            new_record = {
                'viewing_id': f'view_{uuid.uuid4().hex[:9]}',
                # **FIX:** device_id is NOW ONLY generated here
                'device_id': random.choice(profile['device_ids']),
                'ott_platform': random.choice(['Netflix', 'Max', 'Disney+', 'Hulu']),
                'hours_watched_last_month': round(random.uniform(5, 60), 1)
            }
            if random.random() > 0.4:
                new_record['email'] = random.choice(profile['emails'])

        records[source].append(new_record)

    with open('raw_customer_data.json', 'w') as f:
        json.dump(records, f, indent=2)

    total_generated = sum(len(v) for v in records.values())
    print(f"✓ Raw data saved to raw_customer_data.json. Total records generated: {total_generated}")


if __name__ == '__main__':
    generate_realistic_data(num_total_records=10000, num_unique_customers=27)