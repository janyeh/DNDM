import requests
from datetime import datetime
import sys
def send_discord_message(message):
  """Sends a message to the specified Discord channel via webhook.

  Args:
    message: The message content to send.
  """

  webhook_url = "https://discord.com/api/webhooks/1304707589728112700/_2I5ghNrAJeAWYnOJ1kMLScdLt1tGUY75mcq3Aoq_pyk3-sAx-q7x4myxKBerP4_fdAd"
  data = {"content": message}

  response = requests.post(webhook_url, json=data)

  if response.status_code == 204:
    print("Message sent successfully!")

  else:
    print(f"Failed to send message. Status code: {response.status_code}")
    print(response.text)

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = f"{sys.argv[1]}. {datetime.now()}"
    else:
        message = f"Task done. {datetime.now()}"
    send_discord_message(message)