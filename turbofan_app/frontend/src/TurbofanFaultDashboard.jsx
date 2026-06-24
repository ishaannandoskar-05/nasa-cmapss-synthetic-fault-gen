import { useState, useRef, useEffect } from "react";
import * as THREE from "three";
import { Upload, Pause, Play, AlertTriangle, CheckCircle2, Activity } from "lucide-react";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// BUG-017 fix: read from Vite env var so the URL can be changed per-environment
// without touching source code. Set VITE_API_BASE in .env or .env.local.
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:5000/api";

const CLASSES = [
  { id: 0, name: "C0 - Healthy", color: "#1D9E75", hex3d: 0x1d9e75, desc: "No degradation detected" },
  { id: 1, name: "C1 - Early wear", color: "#BA7517", hex3d: 0xba7517, desc: "Subtle efficiency drift" },
  { id: 2, name: "C2 - Advanced fault", color: "#D85A30", hex3d: 0xd85a30, desc: "Clear degradation signature" },
  { id: 3, name: "C3 - Imminent failure", color: "#E24B4A", hex3d: 0xe24b4a, desc: "Critical - immediate action" },
];

const PART_LABELS = {
  fan: "Fan",
  lpc: "LP Compressor",
  hpc: "HP Compressor",
  combustor: "Combustor",
  hpt: "HP Turbine",
  lpt: "LP Turbine",
  nozzle: "Nozzle",
};

const NEUTRAL = 0x9a9890;
const NEUTRAL_DARK = 0x5f5e5a;

// ---------------------------------------------------------------------------
// 3D Engine viewer (vanilla Three.js inside a React-managed canvas)
// ---------------------------------------------------------------------------

function EngineViewer({ faultZones, classColor, autoRotate, onToggleRotate }) {
  const mountRef = useRef(null);
  const stateRef = useRef({});

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const W = mount.clientWidth || 640;
    const H = 360;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 100);
    camera.position.set(0, 1.3, 9);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dir1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dir1.position.set(5, 6, 8);
    scene.add(dir1);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dir2.position.set(-5, -3, -4);
    scene.add(dir2);

    const engineGroup = new THREE.Group();
    engineGroup.rotation.x = 0.08;
    scene.add(engineGroup);

    const partMaterials = {};
    let xCursor = -4.3;

    function makeMaterial(colorHex) {
      return new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: 0x000000,
        emissiveIntensity: 0,
        metalness: 0.4,
        roughness: 0.55,
        transparent: true,
        opacity: 0.92,
      });
    }

    function addStage(name, radiusOuter, length, bladeCount, isBladed) {
      const group = new THREE.Group();
      const mat = makeMaterial(NEUTRAL);
      partMaterials[name] = mat;

      const casingGeo = new THREE.CylinderGeometry(radiusOuter, radiusOuter, length, 32, 1, true);
      const casing = new THREE.Mesh(casingGeo, mat);
      casing.rotation.z = Math.PI / 2;
      group.add(casing);

      if (isBladed) {
        const bladeGeo = new THREE.BoxGeometry(length * 0.82, radiusOuter * 0.92, 0.045);
        for (let i = 0; i < bladeCount; i++) {
          const blade = new THREE.Mesh(bladeGeo, mat);
          const angle = (i / bladeCount) * Math.PI * 2;
          blade.position.set(0, Math.sin(angle) * radiusOuter * 0.46, Math.cos(angle) * radiusOuter * 0.46);
          blade.rotation.x = angle;
          group.add(blade);
        }
      }

      group.position.x = xCursor + length / 2;
      xCursor += length + 0.06;
      engineGroup.add(group);
    }

    addStage("fan", 1.35, 1.05, 18, true);
    addStage("lpc", 0.92, 0.95, 22, true);
    addStage("hpc", 0.74, 0.85, 26, true);
    addStage("combustor", 0.8, 0.95, 0, false);
    addStage("hpt", 0.7, 0.7, 20, true);
    addStage("lpt", 0.78, 0.85, 16, true);
    addStage("nozzle", 0.55, 0.9, 0, false);

    const shaftGeo = new THREE.CylinderGeometry(0.18, 0.18, 8.6, 16);
    const shaft = new THREE.Mesh(shaftGeo, makeMaterial(NEUTRAL_DARK));
    shaft.rotation.z = Math.PI / 2;
    engineGroup.add(shaft);

    let isDragging = false;
    let lastX = 0;
    let lastY = 0;
    renderer.domElement.style.cursor = "grab";

    const onPointerDown = (e) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      renderer.domElement.style.cursor = "grabbing";
    };
    const onPointerUp = () => {
      isDragging = false;
      renderer.domElement.style.cursor = "grab";
    };
    const onPointerMove = (e) => {
      if (!isDragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      engineGroup.rotation.y += dx * 0.008;
      engineGroup.rotation.x += dy * 0.005;
      engineGroup.rotation.x = Math.max(-0.6, Math.min(0.6, engineGroup.rotation.x));
      lastX = e.clientX;
      lastY = e.clientY;
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointermove", onPointerMove);

    let rafId;
    const animate = () => {
      rafId = requestAnimationFrame(animate);
      if (stateRef.current.autoRotate && !isDragging) {
        engineGroup.rotation.y += 0.006;
      }
      renderer.render(scene, camera);
    };
    animate();

    stateRef.current = {
      ...stateRef.current,
      partMaterials,
      renderer,
      scene,
      camera,
      autoRotate,
    };

    const handleResize = () => {
      const newW = mount.clientWidth || 640;
      camera.aspect = newW / H;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, H);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointermove", onPointerMove);
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    stateRef.current.autoRotate = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    const mats = stateRef.current.partMaterials;
    if (!mats) return;
    Object.keys(PART_LABELS).forEach((part) => {
      const mat = mats[part];
      if (!mat) return;
      if (faultZones.includes(part)) {
        mat.color.setHex(classColor);
        mat.emissive.setHex(classColor);
        mat.emissiveIntensity = 0.55;
      } else {
        mat.color.setHex(NEUTRAL);
        mat.emissive.setHex(0x000000);
        mat.emissiveIntensity = 0;
      }
    });
  }, [faultZones, classColor]);

  return (
    <div className="relative w-full rounded-xl overflow-hidden bg-neutral-50">
      <div ref={mountRef} className="w-full" style={{ height: 360 }} />
      <button
        onClick={onToggleRotate}
        aria-label="Toggle rotation"
        className="absolute top-3 right-3 w-9 h-9 rounded-full border border-neutral-200 bg-white/90 flex items-center justify-center hover:bg-white transition-colors shadow-sm"
      >
        {autoRotate ? <Pause size={15} /> : <Play size={15} />}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Probability bar chart
// ---------------------------------------------------------------------------

function ProbabilityBars({ probabilities }) {
  if (!probabilities) return null;
  const entries = Object.entries(probabilities);
  const max = Math.max(...entries.map(([, v]) => v), 0.0001);

  return (
    <div className="flex flex-col gap-2">
      {entries.map(([name, value], i) => (
        <div key={name} className="flex items-center gap-3">
          <span className="text-xs text-neutral-500 w-32 flex-shrink-0">{name}</span>
          <div className="flex-1 h-2 rounded-full bg-neutral-100 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${(value / max) * 100}%`,
                backgroundColor: CLASSES[i].color,
              }}
            />
          </div>
          <span className="text-xs font-medium text-neutral-700 w-12 text-right">
            {(value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main dashboard
// ---------------------------------------------------------------------------

export default function TurbofanFaultDashboard() {
  const [selectedClass, setSelectedClass] = useState(0);
  const [autoRotate, setAutoRotate] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking");
  const fileInputRef = useRef(null);

  useEffect(() => {
    let active = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (active) {
          setApiStatus(data.model_loaded ? "ready" : "model_missing");
        }
      } catch {
        if (active) {
          setApiStatus("offline");
        }
      }
    };
    checkHealth();
    return () => { active = false; };
  }, []);

  const activeClass = prediction ? prediction.predicted_class : selectedClass;
  const activeFaultZones = prediction
    ? prediction.fault_zones
    : { 0: [], 1: ["hpc"], 2: ["hpc", "hpt"], 3: ["hpc", "hpt", "combustor"] }[selectedClass];

  const handleSampleClick = async (classId) => {
    setSelectedClass(classId);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/sample/${classId}`);
      const data = await res.json();
      if (res.ok) {
        setPrediction(data);
      } else {
        setError(data.error || "Failed to fetch sample");
      }
    } catch {
      setError("Could not reach API. Is the Flask backend running on :5000?");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setPrediction(data);
        setSelectedClass(data.predicted_class);
      } else {
        setError(data.error || "Prediction failed");
      }
    } catch {
      setError("Could not reach API. Is the Flask backend running on :5000?");
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const currentClassInfo = CLASSES[activeClass];

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-neutral-900">Turbofan Fault Monitor</h1>
          <p className="text-sm text-neutral-500 mt-1">
            CGAN-synthetic-trained classifier &middot; NASA CMAPSS
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border border-neutral-200">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              apiStatus === "ready"
                ? "bg-emerald-500"
                : apiStatus === "checking"
                ? "bg-amber-400"
                : "bg-red-500"
            }`}
          />
          <span className="text-neutral-600">
            {apiStatus === "ready" && "API connected"}
            {apiStatus === "checking" && "Checking API..."}
            {apiStatus === "model_missing" && "Model not loaded"}
            {apiStatus === "offline" && "API offline"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3 flex flex-col gap-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {CLASSES.map((c) => (
              <button
                key={c.id}
                onClick={() => handleSampleClick(c.id)}
                className="px-3 py-2.5 rounded-lg border text-left transition-colors"
                style={{
                  borderColor: activeClass === c.id ? c.color : "#e5e5e5",
                  backgroundColor: activeClass === c.id ? `${c.color}14` : "white",
                }}
              >
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: c.color }}
                  />
                  <span className="text-xs font-medium text-neutral-800">{c.name}</span>
                </div>
              </button>
            ))}
          </div>

          <EngineViewer
            faultZones={activeFaultZones}
            classColor={currentClassInfo.hex3d}
            autoRotate={autoRotate}
            onToggleRotate={() => setAutoRotate(!autoRotate)}
          />

          <div className="flex items-center gap-2 text-sm text-neutral-600">
            <Activity size={14} className="text-neutral-400" />
            <span>{currentClassInfo.desc}</span>
            {activeFaultZones.length > 0 && (
              <div className="flex gap-1.5 ml-1">
                {activeFaultZones.map((zone) => (
                  <span
                    key={zone}
                    className="text-xs font-medium px-2 py-0.5 rounded-full"
                    style={{
                      backgroundColor: `${currentClassInfo.color}1a`,
                      color: currentClassInfo.color,
                    }}
                  >
                    {PART_LABELS[zone]}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="lg:col-span-2 flex flex-col gap-4">
          <div className="border border-neutral-200 rounded-xl p-4">
            <label className="block text-sm font-medium text-neutral-800 mb-2">
              Upload engine sensor CSV
            </label>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 border border-dashed border-neutral-300 rounded-lg py-6 text-sm text-neutral-500 hover:border-neutral-400 hover:text-neutral-700 transition-colors disabled:opacity-50"
            >
              <Upload size={16} />
              {loading ? "Processing..." : "Click to upload .csv"}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileUpload}
            />
            <p className="text-xs text-neutral-400 mt-2">
              Requires columns: op1, op2, s2, s3, s4, s7, s8, s9, s11-s15, s17, s20, s21
            </p>
          </div>

          {error && (
            <div className="flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-100 rounded-lg p-3">
              <AlertTriangle size={15} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {prediction && (
            <div className="border border-neutral-200 rounded-xl p-4 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {activeClass === 0 ? (
                    <CheckCircle2 size={16} style={{ color: currentClassInfo.color }} />
                  ) : (
                    <AlertTriangle size={16} style={{ color: currentClassInfo.color }} />
                  )}
                  <span className="text-sm font-semibold text-neutral-900">
                    {prediction.class_name}
                  </span>
                </div>
                <span className="text-xs text-neutral-400">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-neutral-50 rounded-lg p-3">
                  <div className="text-xs text-neutral-400 mb-0.5">Est. RUL</div>
                  <div className="font-semibold text-neutral-800">
                    {prediction.rul_estimate} cycles
                  </div>
                </div>
                <div className="bg-neutral-50 rounded-lg p-3">
                  <div className="text-xs text-neutral-400 mb-0.5">Fault zones</div>
                  <div className="font-semibold text-neutral-800">
                    {prediction.fault_zones.length || "None"}
                  </div>
                </div>
              </div>

              <div>
                <div className="text-xs text-neutral-400 mb-2">Class probabilities</div>
                <ProbabilityBars probabilities={prediction.probabilities} />
              </div>
            </div>
          )}

          {!prediction && !error && (
            <div className="text-sm text-neutral-400 text-center py-8 border border-dashed border-neutral-200 rounded-xl">
              Select a class above or upload a CSV to see predictions
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
