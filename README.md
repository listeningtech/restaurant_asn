# Restaurant ASN Demo

Interactive demo page for **Cooperative Node-Specific Speech Enhancement in Resource-Constrained Acoustic Sensor Networks**, accepted at IWAENC 2026.

The demo shows a four-table restaurant acoustic sensor network. For each node, visitors can listen to:

- the noisy microphone mixture,
- the clean local target speech,
- the processed node-specific enhanced output.

## Repository Layout

- `index.html`, `styles.css`, `script.js`: static GitHub Pages demo.
- `assets/audio/`: demo WAV files for the four nodes.
- `assets/figures/`: paper figures used on the demo page.
- `code/`: training, evaluation, plotting, and dataset-generation scripts.

## Local Preview

From this directory:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## GitHub Pages

After creating `listeningtech/restaurant_asn`, push this folder to the repository and enable GitHub Pages from the repository settings using the `main` branch and root directory.

## Citation

Rajesh R, Rashen Fernando, Yuezhong Xu, and Ryan M. Corey, "Cooperative Node-Specific Speech Enhancement in Resource-Constrained Acoustic Sensor Networks," IWAENC 2026.
