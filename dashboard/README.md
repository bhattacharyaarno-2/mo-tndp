# MO-TNDP Dashboard

This folder contains a static dashboard for replaying saved MO-TNDP policies.

## Refresh the data

Run this from the repository root:

```powershell
C:\Users\bhatt\miniconda3\envs\tabular-tndp\python.exe -m motndp.dashboard_data --output dashboard/data.js
```

## Open the UI

Open [index.html](C:\Users\bhatt\Desktop\Thesis\mo-tndp\dashboard\index.html) in a browser.

The dashboard reads `dashboard/data.js`, so you only need to regenerate that file when the saved Q-tables or runs change.
