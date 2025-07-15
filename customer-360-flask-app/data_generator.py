# data_generator.py
import json
import random
import uuid

def generate_realistic_data(num_total_records=10000, num_unique_customers=25):
    """
    Generates a large, messy dataset that converges to a smaller number of unique customers.
    """

    # A richer pool of base profiles to serve as the ground truth
    base_profiles = [
        {'first': 'Jennifer', 'last': 'Mendoza', 'aliases': ['Jen Mendoza', 'J. Mendoza'], 'emails': ['jen.mendoza@gmail.com', 'j.mendoza@workplace.com'], 'phone': '555-0101'},
        {'first': 'Michael', 'last': 'Zhang', 'aliases': ['Mike Zhang', 'Michael Z.'], 'emails': ['mike.zhang@yahoo.com', 'michael.zhang@techcorp.io'], 'phone': '555-0102'},
        {'first': 'Priya', 'last': 'Sharma', 'aliases': ['Priya S.', 'P. Sharma'], 'emails': ['priya.sharma@outlook.com', 'psharma@globalfoods.net'], 'phone': '555-0103'},
        {'first': 'David', 'last': 'Jones', 'aliases': ['Davey Jones', 'David L. Jones'], 'emails': ['daveyjones@gmail.com', 'david.jones@finance.com'], 'phone': '555-0104'},
        {'first': 'Samantha', 'last': 'Rodriguez', 'aliases': ['Sam Rodriguez', 'S. Rodriguez'], 'emails': ['sam.rodriguez@me.com', 'samantha.r@startup.co'], 'phone': '555-0105'},
        {'first': 'Christopher', 'last': 'Lee', 'aliases': ['Chris Lee', 'C. Lee'], 'emails': ['chris.lee@icloud.com', 'christopher.lee@lawfirm.com'], 'phone': '555-0106'},
        {'first': 'Aisha', 'last': 'Khan', 'aliases': ['Aisha K.', 'A. Khan'], 'emails': ['akhan@university.edu', 'aisha.khan@research.org'], 'phone': '555-0107'},
        {'first': 'Daniel', 'last': 'Kim', 'aliases': ['Dan Kim', 'Daniel K.'], 'emails': ['dan.kim@gmail.com', 'dkim@designs.com'], 'phone': '555-0108'},
        {'first': 'Olivia', 'last': 'Chen', 'aliases': ['Liv Chen', 'O. Chen'], 'emails': ['olivia.chen@icloud.com', 'chen.o@pharma.com'], 'phone': '555-0109'},
        {'first': 'William', 'last': 'Taylor', 'aliases': ['Will Taylor', 'Bill Taylor'], 'emails': ['will.taylor@yahoo.com', 'william.t@consulting.com'], 'phone': '555-0110'},
        {'first': 'Sophia', 'last': 'Nguyen', 'aliases': ['Sophie Nguyen', 'S. Nguyen'], 'emails': ['sophia.n@gmail.com', 's.nguyen@fashion.co'], 'phone': '555-0111'},
        {'first': 'James', 'last': 'Brown', 'aliases': ['Jim Brown', 'James B.'], 'emails': ['jamesbrown@live.com', 'j.brown@construction.net'], 'phone': '555-0112'},
        {'first': 'Isabella', 'last': 'Garcia', 'aliases': ['Bella Garcia', 'I. Garcia'], 'emails': ['isabella.g@me.com', 'igarcia@media.com'], 'phone': '555-0113'},
        {'first': 'Benjamin', 'last': 'Martinez', 'aliases': ['Ben Martinez', 'B. Martinez'], 'emails': ['ben.martinez@gmail.com', 'benjamin.m@artstudio.com'], 'phone': '555-0114'},
        {'first': 'Mia', 'last': 'Hernandez', 'aliases': ['Mia H.', 'M. Hernandez'], 'emails': ['mia.hernandez@outlook.com', 'mhernandez@nonprofit.org'], 'phone': '555-0115'},
        {'first': 'Ethan', 'last': 'Lopez', 'aliases': ['E. Lopez'], 'emails': ['ethan.lopez@yahoo.com', 'e.lopez@state.gov'], 'phone': '555-0116'},
        {'first': 'Ava', 'last': 'Gonzalez', 'aliases': ['A. Gonzalez'], 'emails': ['ava.gonzalez@icloud.com', 'gonzalez.ava@hospital.org'], 'phone': '555-0117'},
        {'first': 'Alexander', 'last': 'Wilson', 'aliases': ['Alex Wilson', 'A. Wilson'], 'emails': ['alex.wilson@gmail.com', 'a.wilson@techhub.com'], 'phone': '555-0118'},
        {'first': 'Emily', 'last': 'Anderson', 'aliases': ['Em Anderson', 'E. Anderson'], 'emails': ['emily.a@live.com', 'e.anderson@publishing.com'], 'phone': '555-0119'},
        {'first': 'Jacob', 'last': 'Thomas', 'aliases': ['Jake Thomas', 'J. Thomas'], 'emails': ['jacob.thomas@me.com', 'jthomas@motors.com'], 'phone': '555-0120'},
        {'first': 'Madison', 'last': 'Moore', 'aliases': ['Maddie Moore', 'M. Moore'], 'emails': ['madison.moore@gmail.com', 'mmoore@realestate.com'], 'phone': '555-0121'},
        {'first': 'Matthew', 'last': 'Jackson', 'aliases': ['Matt Jackson', 'M. Jackson'], 'emails': ['matt.jackson@yahoo.com', 'matthew.j@logistics.com'], 'phone': '555-0122'},
        {'first': 'Abigail', 'last': 'White', 'aliases': ['Abby White', 'A. White'], 'emails': ['abby.white@icloud.com', 'a.white@travelco.com'], 'phone': '555-0123'},
        {'first': 'Joshua', 'last': 'Harris', 'aliases': ['Josh Harris', 'J. Harris'], 'emails': ['josh.harris@gmail.com', 'j.harris@sports.com'], 'phone': '555-0124'},
        {'first': 'Chloe', 'last': 'Martin', 'aliases': ['C. Martin'], 'emails': ['chloe.martin@outlook.com', 'cmartin@events.com'], 'phone': '555-0125'}
    ]

    # Ensure we don't try to use more profiles than we have defined
    active_profiles = base_profiles[:num_unique_customers]

    print(f"Generating {num_total_records} records that will resolve to {len(active_profiles)} unique customers...")

    records = {
        'ecommerce': [],
        'mobile_app': [],
        'marketing': []
    }

    for _ in range(num_total_records):
        # Randomly select a base profile and a source system for this new record
        profile = random.choice(active_profiles)
        source = random.choice(['ecommerce', 'mobile_app', 'marketing'])

        new_record = {}

        # Generate a noisy record based on the source
        if source == 'ecommerce':
            new_record = {
                'order_id': f'ord_{uuid.uuid4().hex[:8]}',
                'customer_name': random.choice([f"{profile['first']} {profile['last']}", random.choice(profile['aliases'])]),
                'email': random.choice(profile['emails']),
                'order_total': round(random.uniform(20, 800), 2)
            }
            # 70% chance to include the phone number
            if random.random() > 0.3:
                new_record['phone_number'] = profile['phone']

        elif source == 'mobile_app':
            new_record = {
                'session_id': f'sess_{uuid.uuid4().hex[:10]}',
                'email': random.choice(profile['emails']),
                'time_spent_min': random.randint(1, 180)
            }
            # 80% chance to include a device_id
            if random.random() > 0.2:
                new_record['device_id'] = f"device_{uuid.uuid4().hex[:12]}"

        elif source == 'marketing':
            new_record = {
                'campaign_id': random.choice(['spring_2025', 'q3_promo', 'holiday_push']),
                'email': random.choice(profile['emails']),
                'clicked_ad': random.choice([True, False])
            }
            # 50% chance to include a name
            if random.random() > 0.5:
                new_record['first_name'] = profile['first']
                new_record['last_name'] = profile['last']

        records[source].append(new_record)

    # Save the massive, messy dataset to the JSON file
    with open('raw_customer_data.json', 'w') as f:
        json.dump(records, f, indent=2) # Using indent=2 to save some space

    total_generated = sum(len(v) for v in records.values())
    print(f"✓ Raw data saved to raw_customer_data.json. Total records generated: {total_generated}")


if __name__ == '__main__':
    # You can change the numbers here if you want
    generate_realistic_data(num_total_records=10000, num_unique_customers=25)