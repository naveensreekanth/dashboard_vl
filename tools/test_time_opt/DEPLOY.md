# Deploy (cloud — no local PC required)

This app needs **Node + Python (PyTorch)**. A tunnel through your laptop is only for demos.

## Permanent share link (recommended): Render

1. Open this deploy link (logged into Render with GitHub):

   https://render.com/deploy?repo=https://github.com/mohith1805/Vector-memory-and-time-optimization-

2. Click **Apply** / **Create Web Service**.
3. Wait for the first Docker build (10–20+ minutes; PyTorch image is large).
4. When status is **Live**, copy the URL, e.g.:

   `https://test-time-optimization.onrender.com`

5. Send **only that URL** to your client. They never see your source code.

### Notes

- Use at least the **Starter** plan (free tier often runs out of RAM for PyTorch).
- Keep the GitHub repo **private** if you do not want others cloning the code.
- Fail-log analysis runs in the client’s browser on their own log files.

## Alternative hosts

| Host | Notes |
|------|--------|
| [Railway](https://railway.app) | New project → Deploy from GitHub → select this repo → Docker |
| [Fly.io](https://fly.io) | `fly launch` with the included Dockerfile |
| Any VPS (DigitalOcean, AWS Lightsail) | Install Docker, `docker build` + `docker run -p 80:8787` |

## Local production (your machine only)

```bash
npm run install:all
npm run build
npm start
```

Then open http://localhost:8787
