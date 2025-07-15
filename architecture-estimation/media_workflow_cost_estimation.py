# Python (Illustrative - actual cost estimation requires detailed knowledge of pricing models)

cloud_pricing = {
    "AWS MediaConvert": {"per_minute_HD": 0.01, "per_minute_4K": 0.03}, # Example rates
    "AWS CloudFront": {"per_GB_egress": 0.02}, # Example rate
    "Snowflake": {"compute_per_hour": 3.00, "storage_per_TB_month": 25.00}, # Example rates
    "Databricks": {"premium_instance_hour": 0.50}, # Example rate
    # Add more services and their approximate pricing
}

def estimate_media_workflow_cost(ingest_tb_month=0, streaming_users=0, analytics_frequency="daily"):
    total_cost = 0

    # Rough estimation logic (needs significant refinement based on actual use cases)
    total_cost += ingest_tb_month * cloud_pricing.get("AWS S3", {}).get("storage_per_TB_month", 10) # Assuming some S3 storage
    total_cost += ingest_tb_month * 1024 * cloud_pricing.get("AWS CloudFront", {}).get("per_GB_egress", 0.02) # Rough egress for ingest

    if streaming_users > 0:
        # Very basic assumption - needs much more detail
        total_cost += streaming_users * 0.05 # Placeholder for streaming costs

    if analytics_frequency == "daily":
        total_cost += 30 * cloud_pricing.get("Databricks", {}).get("premium_instance_hour", 0.50) * 4 # Assuming some daily processing

    print("Rough, indicative monthly cost estimate:")
    print(f"Total: ${total_cost:.2f}")
    print("\nNote: This is a highly simplified estimate. Actual costs can vary significantly based on specific usage patterns, instance types, data volumes, and other factors. A detailed cost analysis will be required.")

if __name__ == "__main__":
    ingest = float(input("Enter estimated monthly video ingest volume (in TB): "))
    users = int(input("Enter expected number of concurrent streaming users (if applicable): "))
    frequency = input("Enter data analytics processing frequency (daily/weekly/monthly): ").lower()
    estimate_media_workflow_cost(ingest, users, frequency)