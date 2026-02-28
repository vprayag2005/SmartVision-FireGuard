---
description: How to deploy the SmartVision-FireGuard app to Azure App Service
---

# Azure Deployment Workflow

Follow these steps to deploy your application to Azure using your GitHub Student Developer Pack.

## Prerequisites
1. Your code is pushed to a **GitHub repository**.
2. You have activated your **$100 Azure Credit** from the GitHub Student Developer Pack.

## Step 1: Create Azure Web App
1. Go to the [Azure Portal](https://portal.azure.com).
2. Click **"Create a resource"** and search for **"Web App"**.
3. **Project Details**:
   - **Subscription**: Select your "Azure for Students" subscription.
   - **Resource Group**: Create a new one (e.g., `FireGuard-RG`).
4. **Instance Details**:
   - **Name**: Give your app a unique name (e.g., `smartvision-fireguard-xyz`).
   - **Publish**: `Code`
   - **Runtime stack**: `Python 3.12` (or 3.11)
   - **Operating System**: `Linux`
   - **Region**: Choose the one closest to you (e.g., `East US`).
5. **Pricing Plan**:
   - Select a plan with at least **2.0 GB RAM** (e.g., **Basic B1**). *YOLO AI models need this to run.*

## Step 2: Configure Environment Variables
1. Once the Web App is created, go to **Settings > Configuration**.
2. Click **"New application setting"** for each of these:
   - `SECRET_KEY`: (A random string)
   - `MAIL_USERNAME`: (Your email)
   - `MAIL_PASSWORD`: (Your App Password)
   - `ADMIN_EMAIL`: (Initial admin email)
   - `ADMIN_PASSWORD`: (Initial admin password)
3. Click **Save** and **Continue**.

## Step 3: Connect GitHub
1. Go to **Deployment > Deployment Center**.
2. Select **Source**: `GitHub`.
3. Sign in to GitHub and select your **Repository** and **Branch**.
4. Click **Save**. Azure will now start building your app.

## Step 4: Set Startup Command
1. Go back to **Settings > Configuration**.
2. Click the **"General settings"** tab.
3. In the **"Startup Command"** box, type exactly:
   ```bash
   bash startup.sh
   ```
4. Click **Save**.

## Step 5: Check Logs
1. Go to **Monitoring > Log Stream**.
2. You should see `Gunicorn` starting up and your app running!

---
// turbo-all
