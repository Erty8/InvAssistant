import smtplib
import os
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from config import Config

def save_report_locally(html_content: str, subject: str) -> str:
    """
    Saves the HTML report content to a local directory for archiving and debugging.
    Returns the file path.
    """
    try:
        os.makedirs(Config.REPORTS_DIR, exist_ok=True)
        # Create file name based on current date
        safe_subject = "".join([c if c.isalnum() else "_" for c in subject])
        filename = f"{safe_subject}_{datetime.datetime.now().strftime('%H%M%S')}.html"
        file_path = os.path.join(Config.REPORTS_DIR, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return file_path
    except Exception as e:
        print(f"[Warning] Failed to save copy locally: {e}")
        return ""

def send_portfolio_email(report_content_html: str, custom_subject: str = None) -> bool:
    """
    Sends the generated HTML report to the user's receiver email address using SMTP.
    If SMTP settings are missing or connection fails, it falls back to saving locally
    and printing instructions.
    """
    # 1. Establish Subject Line
    date_str = datetime.datetime.now().strftime("%m/%d/%Y")
    subject = custom_subject or f"Daily Portfolio Insights - {date_str} - US Market Open Update"

    # 2. Check configuration validation
    errors, warnings = Config.validate()
    email_configured = Config.SENDER_EMAIL and Config.RECEIVER_EMAIL and Config.SMTP_PASSWORD

    if not email_configured:
        print("\n[SMTP Notification] SMTP credentials not fully configured. Email dispatch skipped.")
        if Config.SAVE_LOCAL_COPY:
            file_path = save_report_locally(report_content_html, subject)
            if file_path:
                print(f"[SMTP Notification] Saved HTML report locally at: {file_path}")
        return False

    # 3. Assemble standard MIME email structure
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = Config.SENDER_EMAIL
    msg["To"] = Config.RECEIVER_EMAIL

    # Attach both plain text and HTML bodies
    # Simple plain text version
    plain_text = "Please open this email in an HTML-compatible mail client to read the Daily Portfolio Summary."
    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(report_content_html, "html"))

    try:
        # Connect to SMTP Server
        print(f"[SMTP Connection] Connecting to {Config.SMTP_SERVER}:{Config.SMTP_PORT}...")
        if Config.SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(Config.SMTP_SERVER, Config.SMTP_PORT, timeout=15)
        else:
            server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT, timeout=15)
            if Config.SMTP_USE_TLS:
                server.ehlo()
                server.starttls()
                server.ehlo()

        # Login
        if Config.SMTP_USERNAME:
            server.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD)
        else:
            server.login(Config.SENDER_EMAIL, Config.SMTP_PASSWORD)

        # Dispatch
        server.sendmail(Config.SENDER_EMAIL, Config.RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("[SMTP Dispatch] Email sent successfully!")
        
        # Save a copy locally anyway if configured
        if Config.SAVE_LOCAL_COPY:
            save_report_locally(report_content_html, subject)
            
        return True

    except Exception as e:
        print(f"\n[SMTP Error] Failed to send email via SMTP: {e}")
        # Always fallback to local storage if configured or SMTP fails
        if Config.SAVE_LOCAL_COPY:
            file_path = save_report_locally(report_content_html, subject)
            if file_path:
                print(f"[SMTP Error Fallback] Saved report locally at: {file_path}")
        return False
