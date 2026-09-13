# PRISM // Tactical Recon Command (SIH26158 - NTRO)

## 1. 3D Point Cloud Generation
Run the COLMAP automatic reconstructor to generate the 3D model from the drone frames:

```powershell
cd C:\COLMAP
.\COLMAP.bat automatic_reconstructor `
  --workspace_path "C:\Users\pavan\PRISM-Defense-3D\data\workspace" `
  --image_path "C:\Users\pavan\PRISM-Defense-3D\data\workspace\images" `
  --data_type video `
  --quality medium


2. Start the Backend API
Run this in a new terminal to start the telemetry and target data server:

PowerShell
cd C:\Users\pavan\PRISM-Defense-3D\backend
python app.py
3. Start the Frontend Dashboard
Run this in a second terminal to host the Three.js dashboard:

PowerShell
cd C:\Users\pavan\PRISM-Defense-3D
python -m http.server 3000
Open your browser and navigate to http://localhost:3000/frontend/index.html
