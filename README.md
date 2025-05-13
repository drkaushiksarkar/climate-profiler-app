# 🌍 Climate & Health Data Profiler 🩺 - How to Run This App

Hello! This guide will walk you through running the **Climate & Health Data Profiler** application on your own computer (Windows or Mac).

**What is this app?**
This application lets you upload health data (from CSV or Excel files) and climate data (from special GRIB files). It helps you look at the data, understand its quality, see some basic charts and maps, and get ideas on how you might combine the two types of data for analysis. You don't need to write any code!

**Why are we using Docker?**
This app needs some specific tools (like Python and a library called `eccodes` for the climate data) to work correctly. Setting these up manually can be tricky and different on every computer. **Docker** is like a magic box that packages the application *and* all its tools together. You install Docker once, and then you can run this app (and others like it) easily without worrying about complex installations. It makes sure the app runs the same way for everyone.

---

## Step 1: Install Docker Desktop (One-Time Setup)

You only need to do this once. If you already have Docker Desktop installed, you can skip to Step 2.

1.  **Go to the Docker Website:** Open your web browser (like Chrome, Firefox, Edge, or Safari) and go to the official Docker Desktop download page: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2.  **Download Docker Desktop:** Click the download button for your operating system (either **Windows** or **Mac**).
3.  **Install Docker Desktop:**
    * **On Windows:** Double-click the downloaded `.exe` file and follow the installation instructions. It might ask you to enable certain Windows features (like WSL 2) – usually, accepting the defaults is fine. You will likely need to **restart your computer** after installation.
    * **On Mac:** Double-click the downloaded `.dmg` file. Drag the Docker icon into your Applications folder. The first time you run Docker from Applications, it might ask for permission or your password.
4.  **Start Docker Desktop:** After installation (and restarting, if needed), find Docker Desktop in your applications and start it. You should see a small Docker icon (usually a whale) in your system tray (Windows) or menu bar (Mac). It might take a minute or two to start up completely. Make sure the icon indicates Docker is running (it usually turns green or stable).

*Having trouble? Docker's website has detailed installation guides if needed.*

---

## Step 2: Get the Application Files

1.  **Download the Files:** You should have received a folder containing these files:
    * `app.py` (The application code)
    * `requirements.txt` (List of software libraries)
    * `Dockerfile` (Instructions for Docker)
    * `README.md` (This guide)
2.  **Save the Folder:** Save this folder somewhere easy to find on your computer, like your Desktop or Documents folder. Let's call this folder `climate_health_profiler`.

---

## Step 3: Open Your Computer's Terminal (Command Prompt)

This is a window where you can type commands for your computer. Don't worry, you only need to type a few specific commands!

* **On Windows:**
    1.  Click the **Start Menu** (or press the Windows key).
    2.  Type `cmd` or `Command Prompt`.
    3.  Click on the **Command Prompt** application to open it.
* **On Mac:**
    1.  Click the **Spotlight Search** icon (magnifying glass 🔍 in the top-right corner) or press `Command + Spacebar`.
    2.  Type `Terminal`.
    3.  Double-click the **Terminal** application to open it.

You should now see a window with a blinking cursor, waiting for you to type.

---

## Step 4: Go to the Application Folder in the Terminal

You need to tell the terminal where you saved the `climate_health_profiler` folder.

1.  **Type `cd` (which means "change directory"), followed by a space.**
2.  **Drag and Drop the Folder:** Find the `climate_health_profiler` folder you saved in Step 2. Click and drag this folder directly onto the terminal window. The path to the folder should appear automatically after the `cd `.
    * *Example (Windows might look like):* `cd C:\Users\YourName\Desktop\climate_health_profiler`
    * *Example (Mac might look like):* `cd /Users/yourname/Desktop/climate_health_profiler`
3.  **Press Enter/Return:** Hit the Enter (Windows) or Return (Mac) key. Your terminal prompt should change, showing you are now "inside" that folder.

---

## Step 5: Build the Docker "Magic Box" (Image)

Now we tell Docker to build the environment for the app using the `Dockerfile` instructions.

1.  **Make Sure Docker Desktop is Running:** Check that the Docker icon is visible and stable in your system tray/menu bar.
2.  **Type the Build Command:** Carefully type the following command into the terminal window (make sure you are still in the `climate_health_profiler` folder!) and then press Enter/Return:
    ```bash
    docker build -t climate-profiler-app .
    ```
    * `docker build`: Tells Docker to build something.
    * `-t climate-profiler-app`: Gives the "magic box" (image) a simple name: `climate-profiler-app`. You can use this name later.
    * `.`: This single dot tells Docker to look for the `Dockerfile` in the current folder.

3.  **Wait (This Might Take a While!):** Docker will now download the base tools and install all the necessary libraries (including `eccodes`). You'll see lots of text scrolling in the terminal. **This step can take 5-15 minutes or even longer**, especially the very first time you run it, depending on your internet speed. Just let it run until it finishes and you see your normal terminal prompt again. You should see a message near the end saying something like "successfully built..." or "finished".

    *If you see red error messages, double-check that Docker Desktop is running and that you typed the command exactly right.*

---

## Step 6: Run the Application!

Now that the "magic box" (image) is built, you can run the application inside it.

1.  **Type the Run Command:** In the same terminal window, carefully type the following command and press Enter/Return:
    ```bash
    docker run -p 8501:8501 climate-profiler-app
    ```
    * `docker run`: Tells Docker to start a container from an image.
    * `-p 8501:8501`: This connects the application inside the box (running on its internal port 8501) to your computer's port 8501, so your browser can find it.
    * `climate-profiler-app`: The name of the image you built in the previous step.

2.  **App Starts:** You'll see some text appear in the terminal, indicating the application is starting up. It might give you some URLs.

---

## Step 7: Use the Application in Your Browser

1.  **Open Your Web Browser:** Go to Chrome, Firefox, Edge, Safari, etc.
2.  **Go to the Address:** In the address bar, type:
    ```
    http://localhost:8501
    ```
3.  **Press Enter/Return.**
4.  **The Climate & Health Data Profiler app should load!** You can now use the sidebar to upload your files and explore the different analysis tabs.

---

## Step 8: Stopping the Application

1.  **Go Back to the Terminal:** Find the terminal window where you ran the `docker run` command (it will likely still be showing output from the app).
2.  **Press `Ctrl + C`:** Hold down the `Ctrl` key (Control) and press the `C` key once. (On Mac, it's also `Control + C`, not Command+C).
3.  **App Stops:** This will stop the Docker container and the application. You can close the terminal window.

**To run the app again later:** You *don't* need to run the `docker build` command again unless the application files (`app.py`, `requirements.txt`) change. Just repeat **Steps 3, 4, 6, and 7**.

---

That's it! You now have a reliable way to run the Climate & Health Data Profiler using Docker.