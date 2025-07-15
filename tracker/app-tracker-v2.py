import getpass
import win32gui
import win32process
import psutil
import time
import csv
import os
import smtplib
import socket
import pyperclip
import pyautogui
from datetime import datetime
from email.message import EmailMessage
from pywinauto import Application

LOG_FILE = "activity_summary.csv"

def get_active_window():
    """Get the active application and extract browser titles & URLs if applicable."""
    try:
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        process_name = process.name()  # Get process name
        window_title = win32gui.GetWindowText(hwnd)  # Get window title

        # If it's a browser, attempt to get URL
        if process_name in ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"]:
            url = get_browser_url()
            print(window_title, url)
            return f"{process_name} - {window_title} ({url})" if url else f"{process_name} - {window_title}"

        return f"{process_name} - {window_title}" if window_title else process_name
    except Exception as e:
        print(f"Error getting active window: {e}")
        return "Unknown"

def get_browser_url():
    """Extract the active URL from Chrome, Edge, or Firefox without stealing focus."""
    try:
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        process_name = process.name().lower()

        if "chrome" in process_name:
            app = Application(backend="uia").connect(title_re=".*Chrome.*")
            dlg = app.top_window()
            url = dlg.child_window(title="Address and search bar", control_type="Edit").get_value()
            return url

        elif "msedge" in process_name:
            app = Application(backend="uia").connect(title_re=".*Edge.*")
            dlg = app.top_window()
            url = dlg.child_window(title="Address and search bar", control_type="Edit").get_value()
            return url

        elif "firefox" in process_name:
            app = Application(backend="uia").connect(title_re=".*Mozilla Firefox.*")
            dlg = app.top_window()
            url = dlg.child_window(control_type="Document").window_text()
            return url

        elif "brave" in process_name:
            app = Application(backend="uia").connect(title_re=".*Brave.*")
            dlg = app.top_window()
            url = dlg.child_window(title="Address and search bar", control_type="Edit").get_value()
            return url

    except Exception as e:
        print(f"Error getting browser URL: {e}")
        return ""

def track_activity():
    """Continuously track active application usage and summarize time spent per application."""
    app_usage = {}  # Store time spent on each app
    prev_app = get_active_window()
    prev_time = time.time()
    # email_interval = 4 * 60 * 60  # Send email every 4 hours (adjust as needed)
    email_interval = 20
    last_email_time = time.time()

    while True:  # Run forever
        current_app = get_active_window()
        current_time = time.time()

        # Update time spent on previous application
        elapsed_time = current_time - prev_time
        if prev_app in app_usage:
            app_usage[prev_app] += elapsed_time
        else:
            app_usage[prev_app] = elapsed_time

        prev_app = current_app
        prev_time = current_time
        time.sleep(5)  # Check every 5 seconds

        # Check if it's time to send an email
        if time.time() - last_email_time >= email_interval:
            save_summary(app_usage)  # Save current activity to CSV
            send_email_tls()  # Send the email
            # send_email_ssl()
            last_email_time = time.time()  # Reset email timer

def save_summary(app_usage):
    """Save summarized results to CSV."""
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Application", "Time Spent (seconds)"])
        for app, duration in app_usage.items():
            print(app)
            writer.writerow([app, round(duration)])

def send_email_tls():
    """Send email with the logged data using Gmail SMTP with TLS."""
    EMAIL_SENDER = "arincool@gmail.com"
    EMAIL_PASSWORD = "zysxaxxcyvesynrf"  # Use the generated App Password
    EMAIL_RECEIVER = "arindam.bose@ltimindtree.com"

    msg = EmailMessage()
    msg["Subject"] = f"Productivity Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content(f"Attached is the productivity report for {socket.gethostname()} for user - {getpass.getuser()}.")

    with open(LOG_FILE, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="csv", filename="activity_summary.csv")

    # Use TLS instead of SSL
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  # Upgrade connection to TLS
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("Email sent successfully.")


def send_email_ssl():
    """Send email with the logged data using Gmail SMTP with SSL."""
    EMAIL_SENDER = "arincool@gmail.com"
    EMAIL_PASSWORD = "zysxaxxcyvesynrf"  # Use App Password
    EMAIL_RECEIVER = "arindam.bose@ltimindtree.com"

    msg = EmailMessage()
    msg["Subject"] = f"Productivity Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content(f"Attached is the productivity report for {socket.gethostname()} for user - {getpass.getuser()}.")

    with open(LOG_FILE, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="csv", filename="activity_summary.csv")

    # Use SSL instead of TLS (Port 465)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("Email sent successfully.")

# Start tracking in an infinite loop
track_activity()
