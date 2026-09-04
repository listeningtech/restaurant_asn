const nodes = [
  {
    id: 0,
    title: "Node 1",
    tracks: [
      ["Mixture", "Local talkers plus other tables, music, and noise", "mixture", "assets/audio/node0_mixture.wav"],
      ["Clean target", "Only the local table speech at this node", "clean", "assets/audio/node0_clean_target.wav"],
      ["Enhanced output", "Processed node-specific speech estimate", "enhanced", "assets/audio/node0_enhanced.wav"],
    ],
  },
  {
    id: 1,
    title: "Node 2",
    tracks: [
      ["Mixture", "Local talkers plus other tables, music, and noise", "mixture", "assets/audio/node1_mixture.wav"],
      ["Clean target", "Only the local table speech at this node", "clean", "assets/audio/node1_clean_target.wav"],
      ["Enhanced output", "Processed node-specific speech estimate", "enhanced", "assets/audio/node1_enhanced.wav"],
    ],
  },
  {
    id: 2,
    title: "Node 3",
    tracks: [
      ["Mixture", "Local talkers plus other tables, music, and noise", "mixture", "assets/audio/node2_mixture.wav"],
      ["Clean target", "Only the local table speech at this node", "clean", "assets/audio/node2_clean_target.wav"],
      ["Enhanced output", "Processed node-specific speech estimate", "enhanced", "assets/audio/node2_enhanced.wav"],
    ],
  },
  {
    id: 3,
    title: "Node 4",
    tracks: [
      ["Mixture", "Local talkers plus other tables, music, and noise", "mixture", "assets/audio/node3_mixture.wav"],
      ["Clean target", "Only the local table speech at this node", "clean", "assets/audio/node3_clean_target.wav"],
      ["Enhanced output", "Processed node-specific speech estimate", "enhanced", "assets/audio/node3_enhanced.wav"],
    ],
  },
];

const title = document.querySelector("#node-title");
const tracks = document.querySelector("#tracks");
const buttons = Array.from(document.querySelectorAll(".table-node"));

function renderNode(nodeId) {
  const node = nodes.find((item) => item.id === nodeId);
  title.textContent = node.title;
  tracks.replaceChildren();

  for (const [name, description, kind, src] of node.tracks) {
    const card = document.createElement("article");
    card.className = "track";

    const heading = document.createElement("h3");
    const label = document.createElement("span");
    label.textContent = name;
    const pill = document.createElement("span");
    pill.className = `pill ${kind}`;
    pill.textContent = kind;
    heading.append(label, pill);

    const desc = document.createElement("p");
    desc.textContent = description;

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = src;

    card.append(heading, desc, audio);
    tracks.append(card);
  }

  for (const button of buttons) {
    const isActive = Number(button.dataset.node) === nodeId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
    button.querySelector(".node-caption").textContent = isActive ? "local target" : "interferer";
  }
}

for (const button of buttons) {
  button.addEventListener("click", () => renderNode(Number(button.dataset.node)));
}

renderNode(0);
