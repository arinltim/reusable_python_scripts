import torch
from aurora import AuroraSmallPretrained, Batch, Metadata, rollout
from datetime import datetime, timedelta
import numpy as np
import xarray as xr # Useful for handling GRIB/NetCDF data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_aurora_data_for_lat_lon_date(
        latitude: float,
        longitude: float,
        target_date: datetime,
        model_name: str = "aurora-0.25-finetuned.ckpt", # Or "aurora-0.25-small-pretrained.ckpt" for a smaller one
        history_hours: int = 12, # Number of historical hours to provide as input
        forecast_steps: int = 1, # How many forecast steps (e.g., 6-hour steps)
        timestep_hours: int = 6 # Aurora's default timestep
):
    """
    Conceptual function to use Microsoft Aurora for temperature and wind.
    THIS IS HIGHLY EXPERIMENTAL AND REQUIRES REAL METEOROLOGICAL INPUT DATA
    WHICH IS NOT HANDLED BY THIS FUNCTION.

    Args:
        latitude (float): Target latitude.
        longitude (float): Target longitude.
        target_date (datetime): The date and time for which to get the prediction.
                                 This will be mapped to the closest forecast step.
        model_name (str): The specific Aurora checkpoint to load.
        history_hours (int): Number of hours of historical data to provide as input.
                             Aurora typically uses 2 time steps (e.g., current and 6 hours prior).
        forecast_steps (int): Number of forecast steps to roll out. Each step is 6 hours.
        timestep_hours (int): The model's inherent timestep (e.g., 6 for Aurora).

    Returns:
        dict: Predicted temperature and wind for the closest point/time, or error.
    """
    logging.info(f"Attempting to load Aurora model '{model_name}' on CPU...")
    # Instantiate the model. We specify .to('cpu') to ensure it stays on CPU.
    # Note: AuroraHighRes etc. are also available, but larger.
    try:
        model = AuroraSmallPretrained().to('cpu') # Or Aurora() for the larger default
        model.load_checkpoint("microsoft/aurora", model_name)
        model.eval() # Set model to evaluation mode
        logging.info("Aurora model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Aurora model: {e}")
        return {"error": f"Failed to load Aurora model: {e}. Ensure you have downloaded model weights if necessary, and dependencies are met."}

    # --- 1. Define the grid and target location ---
    # Aurora uses a specific global grid (e.g., 0.25 degree resolution)
    # The exact latitude and longitude arrays depend on the model's configuration.
    # For a 0.25 degree resolution, these might look like:
    # lats = np.linspace(90, -90, 721) # (180 / 0.25) + 1
    # lons = np.linspace(0, 359.75, 1440) # (360 / 0.25)

    # Let's use a rough example for the AuroraSmallPretrained which uses 17x32 resolution for random data example.
    # You would need to check the actual metadata of the *fine-tuned* model you're using.
    # For a realistic scenario, these would come from the reanalysis data.

    # Placeholder grid - you MUST replace this with the actual grid of your input data
    # that matches the model's expectations.
    # Example for a global 0.25 degree grid (simplified)
    grid_res = 0.25
    lats_grid = np.arange(90, -90.01, -grid_res)
    lons_grid = np.arange(0, 360, grid_res)

    # Find the closest grid point to the requested lat/long
    closest_lat_idx = np.argmin(np.abs(lats_grid - latitude))
    closest_lon_idx = np.argmin(np.abs(lons_grid - longitude))

    selected_lat = lats_grid[closest_lat_idx]
    selected_lon = lons_grid[closest_lon_idx]
    logging.info(f"Target Lat/Lon ({latitude}, {longitude}) mapped to grid point ({selected_lat}, {selected_lon})")

    # --- 2. Prepare Input Data (THIS IS THE MOST CRITICAL AND MISSING PART) ---
    # This is where you would load actual atmospheric and surface variables
    # from a reanalysis dataset (e.g., ERA5, GFS) for the `history_hours`
    # leading up to your `target_date`.
    # The data needs to be in specific tensor shapes (e.g., Batch, channels, levels, lat, lon).

    # Required variables for Aurora:
    # surf_vars: '2t' (2m temperature), '10u' (10m u-wind), '10v' (10m v-wind), 'msl' (mean sea level pressure)
    # static_vars: 'lsm' (land sea mask), 'z' (geopotential at surface), 'slt' (soil type - often ignored or simplified)
    # atmos_vars: 'z', 'u', 'v', 't', 'q' (geopotential, u-wind, v-wind, temperature, specific humidity) at pressure levels

    # This is a DUMMY BATCH for demonstration. You *must* replace this with real data.
    # The shapes below are for the "AuroraSmallPretrained" example, which is 17x32 resolution.
    # Real data would be much larger (e.g., 721x1440 for 0.25 degree global).

    # Example shapes for a dummy 0.25 degree global input (assuming 2 history steps)
    # (batch_size, num_history_steps, num_lats, num_lons) for surface
    # (batch_size, num_history_steps, num_levels, num_lats, num_lons) for atmospheric

    num_lats_dummy = len(lats_grid) # Should be 721 for 0.25 deg
    num_lons_dummy = len(lons_grid) # Should be 1440 for 0.25 deg
    num_atmos_levels = 13 # Example common levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

    # Dummy historical data (e.g., current time and one past step)
    # You would replace this with actual loaded ERA5 or similar data
    current_time_input = target_date.replace(hour=target_date.hour // timestep_hours * timestep_hours,
                                             minute=0, second=0, microsecond=0)
    history_time_input = current_time_input - timedelta(hours=timestep_hours)

    dummy_surf_vars = {
        '2t': torch.randn(1, 2, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
        '10u': torch.randn(1, 2, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
        '10v': torch.randn(1, 2, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
        'msl': torch.randn(1, 2, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
    }
    dummy_atmos_vars = {
        'z': torch.randn(1, 2, num_atmos_levels, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
        'u': torch.randn(1, 2, num_atmos_levels, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
        'v': torch.randn(1, 2, num_atmos_levels, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
        't': torch.randn(1, 2, num_atmos_levels, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
        'q': torch.randn(1, 2, num_atmos_levels, num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'),
    }
    # Static variables (land-sea mask, surface geopotential, soil type) are constant.
    dummy_static_vars = {
        'lsm': torch.randn(num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'), # Land-sea mask (0 or 1)
        'z': torch.randn(num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'), # Surface geopotential
        'slt': torch.randn(num_lats_dummy, num_lons_dummy, dtype=torch.float32).to('cpu'), # Soil type
    }

    # Metadata for the batch
    batch_metadata = Metadata(
        lat=torch.from_numpy(lats_grid).float(),
        lon=torch.from_numpy(lons_grid).float(),
        time=(history_time_input, current_time_input), # Tuple of history times
        atmos_levels=torch.tensor([
            1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 # Example levels in hPa
        ], dtype=torch.float32)
    )

    initial_batch = Batch(
        surf_vars=dummy_surf_vars,
        static_vars=dummy_static_vars,
        atmos_vars=dummy_atmos_vars,
        metadata=batch_metadata
    )
    logging.warning("Input data is DUMMY. For real results, you MUST replace this with actual meteorological reanalysis data (e.g., ERA5) correctly formatted.")


    # --- 3. Run the model inference ---
    logging.info(f"Starting Aurora rollout for {forecast_steps} steps. This will be very slow on CPU...")
    try:
        with torch.inference_mode(): # Disable gradient calculations for faster inference
            # rollout performs autoregressive forecasting
            predictions = list(rollout(model, initial_batch, steps=forecast_steps))
        logging.info("Aurora inference completed.")
    except Exception as e:
        logging.error(f"Error during Aurora inference: {e}")
        return {"error": f"Error during Aurora inference: {e}. This often means the input data shape or content was incorrect, or memory issues."}

    # --- 4. Extract results for the specific lat/long and target date ---
    # The 'predictions' list contains a Batch object for each forecast step.
    # Each Batch object contains predicted 'surf_vars' and 'atmos_vars'.

    # Find the prediction step closest to the target_date
    closest_prediction_batch = None
    min_time_diff = timedelta.max

    # The last prediction in the list will be the latest forecast step.
    # The time for each prediction step is inferred from initial_batch.metadata.time
    # and the timestep_hours.

    # Calculate the time for each prediction step
    prediction_times = []
    for i in range(forecast_steps):
        # The output of rollout is `(current_time + (i+1) * timestep_hours)`
        pred_time = initial_batch.metadata.time[1] + timedelta(hours=(i + 1) * timestep_hours)
        prediction_times.append(pred_time)

    for i, pred_batch in enumerate(predictions):
        pred_time = prediction_times[i]
        time_diff = abs(target_date - pred_time)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_prediction_batch = pred_batch
            chosen_pred_time = pred_time

    if not closest_prediction_batch:
        return {"error": "Could not find a suitable prediction step for the given date."}

    # Extract data at the closest grid point
    # We need to get the index of the closest_lat_idx and closest_lon_idx from the prediction batch
    # The output tensors have a batch dimension (usually 1), then variable dimensions, then spatial.

    # 2-meter temperature ('2t')
    # Shape: (batch_size, num_history_steps_out, lat, lon) or (batch_size, lat, lon) depending on output
    # For a single prediction step, it's typically (1, lat, lon) for surface vars.
    predicted_temp_2t = closest_prediction_batch.surf_vars.get('2t')
    if predicted_temp_2t is not None:
        # Assuming shape (batch_size, lat, lon) after potential squeeze
        # Or (batch_size, time_steps, lat, lon) then take last time step
        # Let's assume (batch_size, lat, lon) after relevant time step selection.
        if predicted_temp_2t.dim() == 4: # (batch, history_steps, lat, lon)
            # Take the last history step, then squeeze batch dim
            temp_at_point = predicted_temp_2t[0, -1, closest_lat_idx, closest_lon_idx].item()
        elif predicted_temp_2t.dim() == 3: # (batch, lat, lon) if time dim was already squeezed
            temp_at_point = predicted_temp_2t[0, closest_lat_idx, closest_lon_idx].item()
        else:
            temp_at_point = None
            logging.warning("Unexpected '2t' tensor dimension.")
    else:
        temp_at_point = None


    # 10-meter u-wind ('10u') and v-wind ('10v')
    predicted_wind_10u = closest_prediction_batch.surf_vars.get('10u')
    predicted_wind_10v = closest_prediction_batch.surf_vars.get('10v')

    wind_u_at_point = None
    wind_v_at_point = None

    if predicted_wind_10u is not None and predicted_wind_10v is not None:
        if predicted_wind_10u.dim() == 4:
            wind_u_at_point = predicted_wind_10u[0, -1, closest_lat_idx, closest_lon_idx].item()
            wind_v_at_point = predicted_wind_10v[0, -1, closest_lat_idx, closest_lon_idx].item()
        elif predicted_wind_10u.dim() == 3:
            wind_u_at_point = predicted_wind_10u[0, closest_lat_idx, closest_lon_idx].item()
            wind_v_at_point = predicted_wind_10v[0, closest_lat_idx, closest_lon_idx].item()
        else:
            logging.warning("Unexpected '10u' or '10v' tensor dimension.")

        if wind_u_at_point is not None and wind_v_at_point is not None:
            wind_speed = np.sqrt(wind_u_at_point**2 + wind_v_at_point**2)
            wind_direction_deg = (np.degrees(np.arctan2(wind_u_at_point, wind_v_at_point)) + 360) % 360
        else:
            wind_speed = None
            wind_direction_deg = None
    else:
        wind_speed = None
        wind_direction_deg = None

    results = {
        "latitude": selected_lat,
        "longitude": selected_lon,
        "predicted_time_utc": chosen_pred_time.isoformat(),
        "temperature_2m": temp_at_point, # in Kelvin, Aurora predicts in Kelvin
        "wind_speed_10m": wind_speed,    # in m/s
        "wind_direction_10m_deg": wind_direction_deg, # in degrees (0 = North, 90 = East)
        "note": "Temperature is in Kelvin (K). Wind speed is in meters per second (m/s). Wind direction is in degrees (0 = North, 90 = East)."
    }
    return results

if __name__ == "__main__":
    # Charlotte, NC
    target_latitude = 35.23
    target_longitude = -80.84

    # For a forecast, target_date would be in the future.
    # For a "historical" run, target_date would be in the past, but you'd need
    # to feed historical reanalysis data *up to that date* as input.

    # Let's try to get a forecast for tomorrow
    target_dt = datetime.now() + timedelta(days=1, hours=6) # 1 day and 6 hours from now

    print(f"Attempting to get Aurora prediction for Lat: {target_latitude}, Lon: {target_longitude}, Date: {target_dt}")

    # !!! IMPORTANT: This will be SLOW on CPU and the results will be RANDOM
    # because the input data is dummy random data.
    # To get meaningful results, you need to provide real ERA5 or similar data
    # formatted correctly.

    forecast = get_aurora_data_for_lat_lon_date(
        latitude=target_latitude,
        longitude=target_longitude,
        target_date=target_dt,
        forecast_steps=1 # Get one 6-hour forecast step
    )

    print("\n--- Aurora Forecast Result (Experimental with Dummy Data) ---")
    if "error" in forecast:
        print(f"Error: {forecast['error']}")
        print("To make this work, you need to:")
        print("1. Obtain real meteorological data (e.g., ERA5 reanalysis) for the specified time and spatial resolution.")
        print("2. Preprocess that data into the specific tensor formats (surf_vars, atmos_vars, static_vars) that Aurora expects.")
        print("3. Ensure the `metadata` (lat, lon, time, atmos_levels) precisely matches your input data and the model's training.")
        print("4. Be prepared for very long inference times on CPU, or consider using a GPU.")
    else:
        print(f"Target Lat/Lon: ({forecast['latitude']}, {forecast['longitude']})")
        print(f"Predicted Time (UTC): {forecast['predicted_time_utc']}")
        print(f"Temperature (2m): {forecast['temperature_2m']:.2f} K ({forecast['temperature_2m'] - 273.15:.2f} °C)")
        print(f"Wind Speed (10m): {forecast['wind_speed_10m']:.2f} m/s")
        print(f"Wind Direction (10m): {forecast['wind_direction_10m_deg']:.2f}°")
        print(f"Note: {forecast['note']}")