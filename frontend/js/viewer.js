// =================================================================
// PRISM // TACTICAL RECON COMMAND
// VIEWER.JS - 3D POINT-CLOUD INSPECTION, DIGITAL TWIN & INGESTION
// =================================================================

// Global API Base URL (served by FastAPI backend)
const API_BASE = "http://127.0.0.1:8000";

// =================================================================
// 1. LIVE TELEMETRY SYNCHRONIZATION
// =================================================================
let flightData = null;
const video = document.getElementById("uav-video");
const latEl = document.getElementById("lat-val");
const lonEl = document.getElementById("lon-val");
const altEl = document.getElementById("alt-val");
const distEl = document.getElementById("dist-val");
const telemSourceTag = document.getElementById("telem-source-tag");

async function fetchTelemetry() {
    try {
        const res = await fetch(`${API_BASE}/api/telemetry?t=${Date.now()}`);
        if (!res.ok) throw new Error("Telemetry endpoint unavailable");
        flightData = await res.json();
        console.log("Telemetry loaded:", flightData);
        if (telemSourceTag && flightData.source) {
            telemSourceTag.textContent = flightData.source.toUpperCase();
        }
    } catch (err) {
        console.warn("Could not retrieve telemetry from API:", err);
    }
}

if (video) {
    video.addEventListener("timeupdate", () => {
        if (!flightData || !flightData.waypoints || flightData.waypoints.length === 0) return;

        const duration = video.duration || 10;
        const progress = Math.min(Math.max(video.currentTime / duration, 0), 1);
        const index = Math.min(
            Math.floor(progress * flightData.waypoints.length),
            flightData.waypoints.length - 1
        );
        const wp = flightData.waypoints[index];

        if (!wp) return;

        if (latEl) latEl.textContent = Number(wp.latitude).toFixed(6);
        if (lonEl) lonEl.textContent = Number(wp.longitude).toFixed(6);
        if (altEl) altEl.textContent = Number(wp.relative_altitude_m).toFixed(2);
        if (distEl && typeof flightData.total_distance_meters === "number") {
            distEl.textContent = (progress * flightData.total_distance_meters).toFixed(2);
        }

        if (typeof updateVideoOverlay === "function") {
            updateVideoOverlay();
        }
    });
}
fetchTelemetry();

// =================================================================
// 2. THREE.JS VIEWPORT & SCENE SETUP
// =================================================================
const container = document.getElementById("canvas-container");
if (!container) {
    console.error("ERROR: #canvas-container was not found.");
    throw new Error("#canvas-container does not exist.");
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x040609);

const camera = new THREE.PerspectiveCamera(
    55,
    container.clientWidth / container.clientHeight,
    0.01,
    5000
);
camera.position.set(0, 5, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.domElement.style.display = "block";
renderer.domElement.style.width = "100%";
renderer.domElement.style.height = "100%";
renderer.domElement.style.cursor = "grab";
container.appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 0.95);
dirLight.position.set(20, 40, 20);
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0xdbeafe, 0.35);
fillLight.position.set(-20, -10, -20);
scene.add(fillLight);

// =================================================================
// 3. GLOBAL TACTICAL STATE & ORBIT CONTROLS
// =================================================================
let tacticalModelGroup = new THREE.Group();
scene.add(tacticalModelGroup);

const roadPotholeGroup = new THREE.Group();
tacticalModelGroup.add(roadPotholeGroup);
let roadPotholeMarkers = [];
let isRoadPotholesActive = false;
let roadAuditData = null;
let isVideoOverlayActive = true;
let selectedPotholeId = null;

let pointCloud = tacticalModelGroup; // Points to the active tactical model for global transforms
let solidMesh = null;
let pointsNode = null;
let activeShadingMode = "mesh";
let modelHasFaces = false;
let standardMaterial = null;
let wireframeMaterial = null;
let clayMaterial = null;
let pointsMaterial = null;

let selectedPoint = null;
let focusPoint = new THREE.Vector3(0, 0, 0);
let initialCameraPosition = new THREE.Vector3(0, 5, 8);
let initialCameraTarget = new THREE.Vector3(0, 0, 0);
let initialModelPosition = new THREE.Vector3();
let initialModelQuaternion = new THREE.Quaternion();

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enableRotate = true;
controls.enableZoom = true;
controls.enablePan = true;

// Mouse bindings: Left = Orbit, Middle/Right = Pan
controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
controls.mouseButtons.MIDDLE = THREE.MOUSE.PAN;
controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
controls.touches.ONE = THREE.TOUCH.ROTATE;
controls.touches.TWO = THREE.TOUCH.DOLLY_PAN;
controls.target.copy(focusPoint);

// Expose tactical twin context globally for modules
window.tacticalTwin = {
    scene,
    camera,
    renderer,
    controls,
    getPointCloud: () => pointCloud
};

// =================================================================
// 4. CAMERA FOCUS ANIMATION
// =================================================================
let focusAnimation = null;

function focusOnPoint(point, smooth = true) {
    if (!point) return;
    selectedPoint = point.clone();
    focusPoint.copy(point);

    if (!smooth) {
        controls.target.copy(point);
        return;
    }

    const direction = new THREE.Vector3().subVectors(camera.position, controls.target);
    let distance = direction.length();
    if (!Number.isFinite(distance) || distance < 0.5) distance = 1;
    direction.normalize();

    const targetDistance = Math.max(distance * 0.45, 0.5);
    const targetCameraPosition = point.clone().add(direction.multiplyScalar(targetDistance));

    const startCameraPosition = camera.position.clone();
    const startTarget = controls.target.clone();
    const startTime = performance.now();
    const duration = 550;

    if (focusAnimation !== null) cancelAnimationFrame(focusAnimation);

    function animateFocus(now) {
        const elapsed = now - startTime;
        let t = Math.min(elapsed / duration, 1);
        t = 1 - Math.pow(1 - t, 3); // Cubic Ease-out

        camera.position.lerpVectors(startCameraPosition, targetCameraPosition, t);
        controls.target.lerpVectors(startTarget, point, t);
        controls.update();

        if (t < 1) {
            focusAnimation = requestAnimationFrame(animateFocus);
        } else {
            focusAnimation = null;
        }
    }
    focusAnimation = requestAnimationFrame(animateFocus);
}

function resetCamera() {
    if (focusAnimation !== null) {
        cancelAnimationFrame(focusAnimation);
        focusAnimation = null;
    }
    camera.position.copy(initialCameraPosition);
    controls.target.copy(initialCameraTarget);
    focusPoint.copy(initialCameraTarget);
    selectedPoint = null;
    controls.update();
}

// =================================================================
// 5. RAYCASTER & INTERACTION DETECTION
// =================================================================
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.12;
const mouse = new THREE.Vector2();

function updateMousePosition(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function getPointUnderMouse(event) {
    if (!tacticalModelGroup || tacticalModelGroup.children.length === 0) return null;
    updateMousePosition(event);
    raycaster.setFromCamera(mouse, camera);
    const intersections = raycaster.intersectObject(tacticalModelGroup, true);
    if (!intersections || intersections.length === 0 || !intersections[0].point) return null;
    return intersections[0].point.clone();
}

let pointerDownX = 0;
let pointerDownY = 0;
let pointerMoved = false;
const DRAG_THRESHOLD = 5;

renderer.domElement.addEventListener("pointerdown", (event) => {
    pointerDownX = event.clientX;
    pointerDownY = event.clientY;
    pointerMoved = false;
    renderer.domElement.style.cursor = "grabbing";
});

renderer.domElement.addEventListener("pointermove", (event) => {
    const dx = event.clientX - pointerDownX;
    const dy = event.clientY - pointerDownY;
    if (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD) {
        pointerMoved = true;
    }
});

renderer.domElement.addEventListener("pointerup", () => {
    renderer.domElement.style.cursor = "grab";
});

// Double click = inspect & focus point
renderer.domElement.addEventListener("dblclick", (event) => {
    if (measureMode) return;
    const point = getPointUnderMouse(event);
    if (!point) return;
    focusOnPoint(point, true);
});

// =================================================================
// 6. TACTICAL METRIC MEASUREMENT TOOL (SCALED & DECOMPOSED)
// =================================================================
let measureMode = false;
let measurePoints = [];
let tempMarkers = [];
function toggleCaliperMode() {
    measureMode = !measureMode;
    console.log(`[TACTICAL HUD] Measurement Mode: ${measureMode ? "ARMED" : "STANDBY"}`);
    showToast(measureMode ? "CALIPERS ARMED (Click 2 points in 3D viewport)" : "MEASUREMENT STANDBY");
    if (!measureMode) clearMeasurementMarkers();
}

window.addEventListener("keydown", (e) => {
    const activeElement = document.activeElement;
    if (activeElement && (activeElement.tagName === "INPUT" || activeElement.tagName === "TEXTAREA")) return;

    if (e.ctrlKey || e.metaKey) {
        if (e.key === "s" || e.key === "S") {
            e.preventDefault();
            openSaveBaselineModal();
            return;
        }
        if (e.key === "o" || e.key === "O") {
            e.preventDefault();
            const btn = document.getElementById("open-ingest-btn");
            if (btn) btn.click();
            return;
        }
        if (e.key === "d" || e.key === "D") {
            e.preventDefault();
            openCompareModal();
            return;
        }
        if (e.key === "l" || e.key === "L") {
            e.preventDefault();
            const btn = document.getElementById("menu-file-load-baseline");
            if (btn) btn.click();
            return;
        }
    }

    if (e.key === "m" || e.key === "M") {
        toggleCaliperMode();
    }
    if (e.key === "h" || e.key === "H") {
        openHeightRestrictionModal();
    }

    if (e.key === "f" || e.key === "F") {
        if (selectedPoint && !measureMode) focusOnPoint(selectedPoint, true);
    }
    if (e.key === "r" || e.key === "R") {
        resetCamera();
        showToast("Camera recentered to scene.");
    }
});

function clearMeasurementMarkers() {
    measurePoints = [];
    tempMarkers.forEach(m => scene.remove(m));
    tempMarkers = [];
    const measureBox = document.getElementById("measure-readout");
    if (measureBox) measureBox.textContent = "STANDBY";
}

container.addEventListener("click", (event) => {
    if (!measureMode || !pointCloud || pointerMoved) return;

    const point = getPointUnderMouse(event);
    if (!point) return;

    measurePoints.push(point);

    // Drop tactical beacon
    const markerGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0xff0055 });
    const marker = new THREE.Mesh(markerGeo, markerMat);
    marker.position.copy(point);
    scene.add(marker);
    tempMarkers.push(marker);

    // Calculate metrics once two beacons are placed
    if (measurePoints.length === 2) {
        const p1 = measurePoints[0];
        const p2 = measurePoints[1];

        // 1. Raw spatial measurements
        const rawDistance = p1.distanceTo(p2);
        const rawHeight = Math.abs(p2.y - p1.y);

        // 2. Compute invariant scale ratio against flight telemetry
        let metricScale = 1.0;
        if (flightData && typeof flightData.metric_scale_factor === "number" && flightData.metric_scale_factor > 0) {
            // Ground-truth camera trajectory scale (invariant to dense quality preset)
            metricScale = flightData.metric_scale_factor;
        } else {
            // Fallback to bounding box ratio
            const modelBox = new THREE.Box3().setFromObject(pointCloud);
            const modelSize = modelBox.getSize(new THREE.Vector3());
            const modelLength = Math.max(modelSize.x, modelSize.z) || 1.0;
            const realWorldFlightDistance = (flightData && flightData.total_distance_meters) ? flightData.total_distance_meters : 125.0;
            metricScale = realWorldFlightDistance / modelLength;
        }

        // 3. Decompose 3D distance into orthogonal components
        const distanceM = rawDistance * metricScale;
        const heightDiffM = rawHeight * metricScale;
        const horizontalSpanM = Math.sqrt(Math.max(0, (distanceM * distanceM) - (heightDiffM * heightDiffM)));

        console.log(`----------------------------------------`);
        console.log(`SCALE MULTIPLIER: ${metricScale.toFixed(4)}`);
        console.log(`LINE DISTANCE  : ${distanceM.toFixed(2)} m`);
        console.log(`VERT ELEVATION : ${heightDiffM.toFixed(2)} m`);
        console.log(`HORIZ SPAN     : ${horizontalSpanM.toFixed(2)} m`);
        console.log(`----------------------------------------`);

        // Draw connecting line
        const lineMat = new THREE.LineBasicMaterial({ color: 0xff0055, linewidth: 2 });
        const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const caliperLine = new THREE.Line(lineGeo, lineMat);
        scene.add(caliperLine);
        tempMarkers.push(caliperLine);

        // Update HUD display box
        const measureBox = document.getElementById("measure-readout");
        if (measureBox) {
            measureBox.innerHTML = `
                LINE: <b>${distanceM.toFixed(2)}m</b><br>
                VERT (H): <b>${heightDiffM.toFixed(2)}m</b><br>
                HORIZ (D): <b>${horizontalSpanM.toFixed(2)}m</b>
            `;
        }

        measurePoints = [];
    }
});

// =================================================================
// 7. AUTOMATIC POINT-CLOUD ALIGNMENT (PCA COVARIANCE)
// =================================================================
function autoAlignPointCloud(points) {
    if (!points || points.length < 3) {
        const fallback = new THREE.Quaternion();
        fallback.setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
        return fallback;
    }

    const center = new THREE.Vector3();
    for (const point of points) center.add(point);
    center.divideScalar(points.length);

    let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (const point of points) {
        const x = point.x - center.x;
        const y = point.y - center.y;
        const z = point.z - center.z;
        xx += x * x; xy += x * y; xz += x * z;
        yy += y * y; yz += y * z; zz += z * z;
    }

    const n = points.length;
    xx /= n; xy /= n; xz /= n; yy /= n; yz /= n; zz /= n;

    const matrix = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
    const vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

    for (let iteration = 0; iteration < 30; iteration++) {
        let p = 0, q = 1, max = Math.abs(matrix[0][1]);
        if (Math.abs(matrix[0][2]) > max) { p = 0; q = 2; max = Math.abs(matrix[0][2]); }
        if (Math.abs(matrix[1][2]) > max) { p = 1; q = 2; max = Math.abs(matrix[1][2]); }
        if (max < 1e-10) break;

        const angle = 0.5 * Math.atan2(2 * matrix[p][q], matrix[q][q] - matrix[p][p]);
        const c = Math.cos(angle), s = Math.sin(angle);
        const mpp = matrix[p][p], mqq = matrix[q][q], mpq = matrix[p][q];

        matrix[p][p] = c * c * mpp - 2 * s * c * mpq + s * s * mqq;
        matrix[q][q] = s * s * mpp + 2 * s * c * mpq + c * c * mqq;
        matrix[p][q] = 0; matrix[q][p] = 0;

        for (let k = 0; k < 3; k++) {
            if (k === p || k === q) continue;
            const mkp = matrix[k][p], mkq = matrix[k][q];
            matrix[k][p] = c * mkp - s * mkq; matrix[p][k] = matrix[k][p];
            matrix[k][q] = s * mkp + c * mkq; matrix[q][k] = matrix[k][q];
        }

        for (let k = 0; k < 3; k++) {
            const vkp = vectors[k][p], vkq = vectors[k][q];
            vectors[k][p] = c * vkp - s * vkq;
            vectors[k][q] = s * vkp + c * vkq;
        }
    }

    const eigenvalues = [matrix[0][0], matrix[1][1], matrix[2][2]];
    const eigenvectors = [
        new THREE.Vector3(vectors[0][0], vectors[1][0], vectors[2][0]),
        new THREE.Vector3(vectors[0][1], vectors[1][1], vectors[2][1]),
        new THREE.Vector3(vectors[0][2], vectors[1][2], vectors[2][2])
    ];

    const indices = [0, 1, 2].sort((a, b) => eigenvalues[b] - eigenvalues[a]);
    const mainDirection = eigenvectors[indices[0]].clone().normalize();
    const upDirection = eigenvectors[indices[2]].clone().normalize();

    if (upDirection.y < 0) upDirection.negate();
    if (mainDirection.z < 0) mainDirection.negate();

    mainDirection.projectOnPlane(upDirection);
    if (mainDirection.lengthSq() < 1e-10) {
        mainDirection.set(0, 0, 1).projectOnPlane(upDirection);
    }
    mainDirection.normalize();

    const rightDirection = new THREE.Vector3().crossVectors(mainDirection, upDirection).normalize();
    const correctedMain = new THREE.Vector3().crossVectors(upDirection, rightDirection).normalize();

    const rotationMatrix = new THREE.Matrix4().makeBasis(rightDirection, upDirection, correctedMain);
    return new THREE.Quaternion().setFromRotationMatrix(rotationMatrix);
}

// =================================================================
// 8. MODEL LOADER & DYNAMIC RELOAD ENGINE
// =================================================================
const loader = new THREE.PLYLoader();

// Persistent Coordinate Frame Anchor (ensures 100% position lock between Points and Mesh)
let savedModelCenter = new THREE.Vector3();
let savedModelQuaternion = new THREE.Quaternion();
let savedModelPositionY = 0;
let hasSavedAlignment = false;

function updateModelTag(customPath = null) {
    const modelTag = document.getElementById("model-tag");
    if (!modelTag) return;
    if (typeof currentDisplayedBaseline !== "undefined" && currentDisplayedBaseline) {
        const ptCount = pointsNode && pointsNode.geometry && pointsNode.geometry.attributes.position
            ? pointsNode.geometry.attributes.position.count.toLocaleString() : "Dense";
        const bName = (currentDisplayedBaseline.name || currentDisplayedBaseline.id).toUpperCase();
        modelTag.textContent = `${ptCount} 3D POINTS | BASELINE: ${bName}`;
        modelTag.style.borderColor = "var(--accent-amber)";
        modelTag.style.color = "var(--accent-amber)";
        return;
    }
    if (customPath && customPath.includes("_diff")) {
        const ptCount = pointsNode && pointsNode.geometry && pointsNode.geometry.attributes.position
            ? pointsNode.geometry.attributes.position.count.toLocaleString() : "Dense";
        modelTag.textContent = `${ptCount} 3D POINTS | TEMPORAL CHANGE MAP (CRIMSON HIGHLIGHT)`;
        modelTag.style.borderColor = "var(--accent-red)";
        modelTag.style.color = "var(--accent-red)";
        return;
    }
    if (activeShadingMode === "points" || !modelHasFaces || !solidMesh) {
        const ptCount = pointsNode && pointsNode.geometry && pointsNode.geometry.attributes.position
            ? pointsNode.geometry.attributes.position.count.toLocaleString() : "Dense";
        modelTag.textContent = `${ptCount} 3D POINTS (TRUE POINT CLOUD)`;
        modelTag.style.borderColor = "var(--panel-border)";
        modelTag.style.color = "var(--accent-cyan)";
    } else if (solidMesh && solidMesh.geometry) {
        const vertCount = solidMesh.geometry.attributes.position
            ? solidMesh.geometry.attributes.position.count.toLocaleString() : "--";
        const faceCount = solidMesh.geometry.index
            ? (solidMesh.geometry.index.count / 3).toLocaleString() : "--";
        modelTag.textContent = `${vertCount} VERTS | ${faceCount} FACES (BLENDER 3D MESH)`;
        modelTag.style.borderColor = "var(--accent-green)";
        modelTag.style.color = "var(--accent-green)";
    }
}

function applyShadingMode(mode) {
    activeShadingMode = mode;
    const viewButtons = {
        points: document.getElementById("view-mode-points"),
        mesh: document.getElementById("view-mode-mesh"),
        wireframe: document.getElementById("view-mode-wireframe"),
        clay: document.getElementById("view-mode-clay")
    };

    Object.keys(viewButtons).forEach(k => {
        if (viewButtons[k]) {
            viewButtons[k].classList.toggle("active", k === mode);
        }
    });

    // Update Format menu checkmarks and active states
    const formatKeys = ["points", "mesh", "wireframe", "clay"];
    formatKeys.forEach(k => {
        const checkEl = document.getElementById(`check-format-${k}`);
        if (checkEl) checkEl.textContent = (k === mode) ? "✓" : "";
        const itemEl = document.getElementById(`menu-format-${k}`);
        if (itemEl) itemEl.classList.toggle("active", k === mode);
    });

    // Sync desktop status bar readout
    const statusMode = document.getElementById("status-shading-mode");
    if (statusMode) {
        statusMode.textContent = `FORMAT: ${mode.toUpperCase()}`;
    }

    if (mode === "points") {
        if (pointsNode) pointsNode.visible = true;
        if (solidMesh) solidMesh.visible = false;
    } else {
        if (solidMesh) {
            if (pointsNode) pointsNode.visible = false;
            solidMesh.visible = true;
            if (mode === "mesh") {
                solidMesh.material = standardMaterial;
            } else if (mode === "wireframe") {
                solidMesh.material = wireframeMaterial;
            } else if (mode === "clay") {
                solidMesh.material = clayMaterial;
            }
        } else {
            if (pointsNode) pointsNode.visible = true;
            showToast("Active model is Point Cloud. Select 'File -> Meshify Surface' to synthesize 3D surfaces!");
        }
    }

    updateModelTag();
}

function loadActiveModel(onComplete, customPath = "data/models/actionable_threat_map.ply") {
    const isCustomDiff = customPath && customPath.includes("_diff");
    const isBaseline = customPath && customPath.includes("baselines");
    const resolvedPointsPath = (isCustomDiff || isBaseline) ? customPath : "data/models/actionable_threat_map_points.ply";
    const plyPath = `${API_BASE}/${resolvedPointsPath}?t=${Date.now()}`;
    console.log("Loading 3D Tactical Model from:", plyPath);

    loader.load(
        plyPath,
        (geometry) => {
            console.log("Point Cloud PLY loaded successfully.");

            // Clear old children in tacticalModelGroup (preserving roadPotholeGroup)
            for (let i = tacticalModelGroup.children.length - 1; i >= 0; i--) {
                const obj = tacticalModelGroup.children[i];
                if (obj === roadPotholeGroup) continue;
                tacticalModelGroup.remove(obj);
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) {
                    if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                    else obj.material.dispose();
                }
            }
            if (!tacticalModelGroup.children.includes(roadPotholeGroup)) {
                tacticalModelGroup.add(roadPotholeGroup);
            }

            geometry.computeVertexNormals();
            geometry.computeBoundingBox();

            const positionAttribute = geometry.getAttribute("position");
            const hasColors = geometry.hasAttribute("color");

            // Build Materials
            pointsMaterial = new THREE.PointsMaterial({
                size: 0.08,
                vertexColors: hasColors
            });

            standardMaterial = new THREE.MeshStandardMaterial({
                vertexColors: hasColors,
                roughness: 0.55,
                metalness: 0.1,
                side: THREE.DoubleSide
            });

            wireframeMaterial = new THREE.MeshBasicMaterial({
                color: 0x64748b,
                wireframe: true
            });

            clayMaterial = new THREE.MeshStandardMaterial({
                color: 0xc4ccd4,
                roughness: 0.45,
                metalness: 0.05,
                side: THREE.DoubleSide
            });

            // Anchor point cloud geometry to origin and remember exact translation offset
            const center = new THREE.Vector3();
            geometry.boundingBox.getCenter(center);
            savedModelCenter.copy(center);
            geometry.translate(-savedModelCenter.x, -savedModelCenter.y, -savedModelCenter.z);

            pointsNode = new THREE.Points(geometry, pointsMaterial);
            tacticalModelGroup.add(pointsNode);
            solidMesh = null;
            modelHasFaces = false;

            pointCloud = tacticalModelGroup;

            const samplePoints = [];
            if (positionAttribute) {
                const total = positionAttribute.count;
                const step = Math.max(1, Math.floor(total / 5000));
                for (let i = 0; i < total; i += step) {
                    samplePoints.push(
                        new THREE.Vector3(
                            positionAttribute.getX(i),
                            positionAttribute.getY(i),
                            positionAttribute.getZ(i)
                        )
                    );
                }
            }

            savedModelQuaternion.copy(autoAlignPointCloud(samplePoints));
            tacticalModelGroup.quaternion.copy(savedModelQuaternion);
            tacticalModelGroup.position.set(0, 0, 0);
            tacticalModelGroup.updateMatrixWorld(true);

            const modelBox = new THREE.Box3().setFromObject(tacticalModelGroup);
            const modelCenter = modelBox.getCenter(new THREE.Vector3());

            // Center vertically
            savedModelPositionY = -modelCenter.y;
            tacticalModelGroup.position.y = savedModelPositionY;
            tacticalModelGroup.updateMatrixWorld(true);

            const finalBox = new THREE.Box3().setFromObject(tacticalModelGroup);
            const finalCenter = finalBox.getCenter(new THREE.Vector3());
            const finalSize = finalBox.getSize(new THREE.Vector3());

            initialModelPosition.copy(tacticalModelGroup.position);
            initialModelQuaternion.copy(tacticalModelGroup.quaternion);

            const maxDimension = Math.max(finalSize.x, finalSize.y, finalSize.z);
            const cameraDistance = Math.max(maxDimension * 1.5, 5);

            camera.position.set(
                finalCenter.x,
                finalCenter.y + cameraDistance * 0.45,
                finalCenter.z + cameraDistance
            );

            focusPoint.copy(finalCenter);
            controls.target.copy(finalCenter);
            controls.update();

            initialCameraPosition.copy(camera.position);
            initialCameraTarget.copy(finalCenter);

            // Default view to points
            if (!isCustomDiff) {
                activeShadingMode = "points";
            }
            applyShadingMode(activeShadingMode);

            // Check if pre-computed solid mesh exists and mount it seamlessly
            if (!isCustomDiff && !isBaseline) {
                fetch(`${API_BASE}/data/models/actionable_threat_mesh.ply?t=${Date.now()}`, { method: "HEAD" })
                    .then(headRes => {
                        if (headRes.ok) {
                            loader.load(
                                `${API_BASE}/data/models/actionable_threat_mesh.ply?t=${Date.now()}`,
                                (meshGeo) => {
                                    meshGeo.translate(-savedModelCenter.x, -savedModelCenter.y, -savedModelCenter.z);
                                    meshGeo.computeVertexNormals();
                                    solidMesh = new THREE.Mesh(meshGeo, standardMaterial);
                                    solidMesh.visible = (activeShadingMode !== "points");
                                    tacticalModelGroup.add(solidMesh);
                                    modelHasFaces = true;
                                    updateModelTag();
                                }
                            );
                        }
                    })
                    .catch(() => {});
            }

            console.log(`Tactical 3D Model mounted.`);
            updateModelTag(customPath);

            if (onComplete) onComplete();
        },
        (xhr) => {
            if (xhr.total) {
                const percent = (xhr.loaded / xhr.total) * 100;
                console.log(`PLY loading: ${percent.toFixed(1)}%`);
            }
        },
        (error) => {
            if (resolvedPointsPath !== "data/models/actionable_threat_map.ply" && !isCustomDiff && !isBaseline) {
                console.warn("Retrying with actionable_threat_map.ply fallback...");
                loadActiveModel(onComplete, "data/models/actionable_threat_map.ply");
            } else {
                console.warn("Could not load PLY model:", error);
            }
        }
    );
}

// Setup Viewport Shading Mode Event Listeners
const viewModePoints = document.getElementById("view-mode-points");
const viewModeMesh = document.getElementById("view-mode-mesh");
const viewModeWireframe = document.getElementById("view-mode-wireframe");
const viewModeClay = document.getElementById("view-mode-clay");

if (viewModePoints) viewModePoints.addEventListener("click", () => applyShadingMode("points"));
if (viewModeMesh) viewModeMesh.addEventListener("click", () => applyShadingMode("mesh"));
if (viewModeWireframe) viewModeWireframe.addEventListener("click", () => applyShadingMode("wireframe"));
if (viewModeClay) viewModeClay.addEventListener("click", () => applyShadingMode("clay"));

// Export OBJ Button
function triggerExportOBJ() {
    showToast("Initiating Blender OBJ model export...");
    const url = currentDisplayedBaseline
        ? `${API_BASE}/api/model/download/obj?baseline_id=${encodeURIComponent(currentDisplayedBaseline.id)}`
        : `${API_BASE}/api/model/download/obj`;
    window.open(url, "_blank");
}
const exportObjBtn = document.getElementById("export-obj-btn");
if (exportObjBtn) {
    exportObjBtn.addEventListener("click", triggerExportOBJ);
}

// Clean & Simple Surface Reconstruction & Progress Manager
let meshifyProgressInterval = null;
let meshifyElapsedTimerInterval = null;

async function triggerMeshify() {
    const meshifyMenu = document.getElementById("menu-file-meshify");
    if (meshifyMenu) meshifyMenu.classList.add("loading");

    const modal = document.getElementById("meshify-progress-modal");
    const percentNum = document.getElementById("meshify-percent-num");
    const progressFill = document.getElementById("meshify-progress-fill");
    const stageTitle = document.getElementById("meshify-stage-title");
    const elapsedTimer = document.getElementById("meshify-elapsed-timer");

    // Reset clean modal UI
    if (modal) modal.classList.add("active");
    if (percentNum) {
        percentNum.textContent = "0%";
        percentNum.classList.remove("complete");
    }
    if (progressFill) {
        progressFill.style.width = "0%";
        progressFill.classList.remove("complete");
    }
    if (stageTitle) stageTitle.textContent = "Connecting photogrammetry points...";
    if (elapsedTimer) elapsedTimer.textContent = "00:00.0s";

    const startTime = Date.now();
    if (meshifyElapsedTimerInterval) clearInterval(meshifyElapsedTimerInterval);
    meshifyElapsedTimerInterval = setInterval(() => {
        const elapsedMs = Date.now() - startTime;
        const totalSec = Math.floor(elapsedMs / 1000);
        const tenths = Math.floor((elapsedMs % 1000) / 100);
        const mins = String(Math.floor(totalSec / 60)).padStart(2, "0");
        const secs = String(totalSec % 60).padStart(2, "0");
        if (elapsedTimer) elapsedTimer.textContent = `${mins}:${secs}.${tenths}s`;
    }, 100);

    // Smooth progressive percentage
    let currentPct = 0;
    let targetPct = 15;

    if (meshifyProgressInterval) clearInterval(meshifyProgressInterval);
    meshifyProgressInterval = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;

        if (elapsed < 2.0) {
            targetPct = Math.min(28, Math.floor(elapsed * 14));
            if (stageTitle) stageTitle.textContent = "Connecting photogrammetry points...";
        } else if (elapsed < 6.0) {
            targetPct = Math.min(65, 28 + Math.floor((elapsed - 2.0) * 9.2));
            if (stageTitle) stageTitle.textContent = "Synthesizing sharp surfaces matching video...";
        } else if (elapsed < 10.0) {
            targetPct = Math.min(88, 65 + Math.floor((elapsed - 6.0) * 5.8));
            if (stageTitle) stageTitle.textContent = "Detecting and sealing interior gaps...";
        } else {
            targetPct = Math.min(96, 88 + Math.floor((elapsed - 10.0) * 1.5));
            if (stageTitle) stageTitle.textContent = "Applying photogrammetric colors & normals...";
        }

        if (currentPct < targetPct) {
            currentPct += 1;
            if (percentNum) percentNum.textContent = `${currentPct}%`;
            if (progressFill) progressFill.style.width = `${currentPct}%`;
        }
    }, 100);

    try {
        const payload = currentDisplayedBaseline ? { baseline_id: currentDisplayedBaseline.id } : {};
        const res = await fetch(`${API_BASE}/api/model/meshify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Clear intervals
        if (meshifyProgressInterval) clearInterval(meshifyProgressInterval);
        if (meshifyElapsedTimerInterval) clearInterval(meshifyElapsedTimerInterval);

        if (res.ok && data.success) {
            // Jump smoothly to 100%
            if (percentNum) {
                percentNum.textContent = "100%";
                percentNum.classList.add("complete");
            }
            if (progressFill) {
                progressFill.style.width = "100%";
                progressFill.classList.add("complete");
            }
            if (stageTitle) stageTitle.textContent = "Solid 3D Surface Ready • Gaps Sealed!";

            // Load and mount mesh in Three.js
            const meshUrl = `${API_BASE}/${data.mesh_ply_url}?t=${Date.now()}`;
            loader.load(
                meshUrl,
                (meshGeo) => {
                    meshGeo.translate(-savedModelCenter.x, -savedModelCenter.y, -savedModelCenter.z);
                    meshGeo.computeVertexNormals();
                    if (solidMesh) {
                        tacticalModelGroup.remove(solidMesh);
                        if (solidMesh.geometry) solidMesh.geometry.dispose();
                    }
                    solidMesh = new THREE.Mesh(meshGeo, standardMaterial);
                    tacticalModelGroup.add(solidMesh);
                    modelHasFaces = true;
                    applyShadingMode("mesh");
                    updateModelTag();
                    loadTacticalTargets();
                },
                undefined,
                (loadErr) => {
                    console.error("Could not mount synthesized mesh:", loadErr);
                }
            );

            // Clean, quick dismissal
            setTimeout(() => {
                if (modal) modal.classList.remove("active");
                showToast(`Solid 3D Mesh Synthesized: ${data.vertices.toLocaleString()} vertices, ${data.triangles.toLocaleString()} faces.`);
            }, 1200);

        } else {
            if (stageTitle) stageTitle.textContent = "Synthesis failed";
            showToast(`Mesh synthesis failed: ${data.detail || "Server error"}`);
            setTimeout(() => {
                if (modal) modal.classList.remove("active");
            }, 2000);
        }
    } catch (err) {
        if (meshifyProgressInterval) clearInterval(meshifyProgressInterval);
        if (meshifyElapsedTimerInterval) clearInterval(meshifyElapsedTimerInterval);
        console.error("Meshify error:", err);
        if (stageTitle) stageTitle.textContent = "Connection error";
        showToast(`Error invoking mesh synthesizer: ${err.message}`);
        setTimeout(() => {
            if (modal) modal.classList.remove("active");
        }, 2000);
    } finally {
        if (meshifyMenu) meshifyMenu.classList.remove("loading");
    }
}

const meshifyModalEl = document.getElementById("meshify-progress-modal");
if (meshifyModalEl) {
    meshifyModalEl.addEventListener("click", (e) => {
        const closeBtn = document.getElementById("meshify-close-btn");
        if (e.target === meshifyModalEl && closeBtn && closeBtn.style.display !== "none") {
            meshifyModalEl.classList.remove("active");
        }
    });
}

const meshifyBtn = document.getElementById("meshify-btn");
if (meshifyBtn) {
    meshifyBtn.addEventListener("click", triggerMeshify);
}

// Initial Model Load
loadActiveModel(() => {
    loadTacticalTargets();
});

// Manual Keyboard Calibration Controls
window.addEventListener("keydown", (event) => {
    if (!pointCloud) return;

    const activeElement = document.activeElement;
    if (activeElement && (activeElement.tagName === "INPUT" || activeElement.tagName === "TEXTAREA")) return;

    const moveStep = 0.5;
    const rotationStep = 0.05;
    let modelChanged = false;

    // Translation (Arrows + Q/E)
    if (event.key === "ArrowUp") { pointCloud.position.z -= moveStep; modelChanged = true; }
    if (event.key === "ArrowDown") { pointCloud.position.z += moveStep; modelChanged = true; }
    if (event.key === "ArrowLeft") { pointCloud.position.x -= moveStep; modelChanged = true; }
    if (event.key === "ArrowRight") { pointCloud.position.x += moveStep; modelChanged = true; }
    if (event.key === "q" || event.key === "Q") { pointCloud.position.y += moveStep; modelChanged = true; }
    if (event.key === "e" || event.key === "E") { pointCloud.position.y -= moveStep; modelChanged = true; }

    // Rotation: Pitch (W/S), Roll (A/D), Yaw (Z/C)
    if (event.key === "w" || event.key === "W") { pointCloud.rotation.x += rotationStep; modelChanged = true; }
    if (event.key === "s" || event.key === "S") { pointCloud.rotation.x -= rotationStep; modelChanged = true; }
    if (event.key === "a" || event.key === "A") { pointCloud.rotation.z += rotationStep; modelChanged = true; }
    if (event.key === "d" || event.key === "D") { pointCloud.rotation.z -= rotationStep; modelChanged = true; }
    if (event.key === "z" || event.key === "Z") { pointCloud.rotation.y += rotationStep; modelChanged = true; }
    if (event.key === "c" || event.key === "C") { pointCloud.rotation.y -= rotationStep; modelChanged = true; }

    if (modelChanged) {
        console.log(`Model Calibration -> Pos: [${pointCloud.position.x.toFixed(2)}, ${pointCloud.position.y.toFixed(2)}, ${pointCloud.position.z.toFixed(2)}] | Rot: [${pointCloud.rotation.x.toFixed(2)}, ${pointCloud.rotation.y.toFixed(2)}, ${pointCloud.rotation.z.toFixed(2)}]`);
    }
});

// =================================================================
// 8.5 TACTICAL TARGET MARKERS & BOUNDING BOXES
// =================================================================
const tacticalTargetGroup = new THREE.Group();
scene.add(tacticalTargetGroup);
let targetObjects = [];

const diffAlertGroup = new THREE.Group();
scene.add(diffAlertGroup);
let diffAlertObjects = [];

async function loadTacticalTargets() {
    try {
        const res = await fetch(`${API_BASE}/api/targets`);
        if (!res.ok) return;
        const data = await res.json();

        // Clear existing target objects
        targetObjects.forEach(obj => tacticalTargetGroup.remove(obj));
        targetObjects = [];

        data.targets.forEach(tgt => {
            const [w, h, d] = tgt.dimensions.map(v => v * 0.08);
            const [x, y, z] = tgt.position;

            const boxGeo = new THREE.BoxGeometry(w, h, d);
            const edgeGeo = new THREE.EdgesGeometry(boxGeo);
            const boxColor = tgt.threat_level === "HIGH" ? 0xef4444 : (tgt.threat_level === "ELEVATED" ? 0xf59e0b : 0x3b82f6);
            const boxMat = new THREE.LineBasicMaterial({ color: boxColor, linewidth: 2 });
            const boundingBox = new THREE.LineSegments(edgeGeo, boxMat);
            boundingBox.position.set(x, y, z);

            const beaconGeo = new THREE.ConeGeometry(0.15, 0.4, 4);
            const beaconMat = new THREE.MeshBasicMaterial({ color: boxColor, wireframe: true });
            const beacon = new THREE.Mesh(beaconGeo, beaconMat);
            beacon.rotation.x = Math.PI;
            beacon.position.set(x, y + (h / 2) + 0.35, z);

            const targetObj = new THREE.Group();
            targetObj.add(boundingBox);
            targetObj.add(beacon);
            targetObj.userData = tgt;

            tacticalTargetGroup.add(targetObj);
            targetObjects.push(targetObj);
        });
        console.log(`Loaded ${data.targets.length} tactical targets.`);
    } catch (err) {
        console.warn("Could not load tactical targets:", err);
    }
}

// Click to inspect target or 3D change alert
container.addEventListener("click", (event) => {
    if (measureMode || pointerMoved) return;
    updateMousePosition(event);
    raycaster.setFromCamera(mouse, camera);

    // 1. Check diff structural change alerts
    const diffHits = raycaster.intersectObjects(diffAlertGroup.children, true);
    if (diffHits.length > 0) {
        let root = diffHits[0].object;
        while (root.parent && root.parent !== diffAlertGroup) {
            root = root.parent;
        }
        if (root.userData && root.userData.id) {
            const data = root.userData;
            const measureBox = document.getElementById("measure-readout");
            if (measureBox) {
                measureBox.innerHTML = `
                    ALERT: <b style="color:#ff3366">${data.id}</b><br>
                    TYPE: <b>${data.type}</b><br>
                    HEIGHT &Delta;H: <b style="color:#ff3366">+${data.max_height_gain_m}m</b><br>
                    FOOTPRINT: <b>${data.footprint_m2} m²</b>
                `;
            }
            focusOnPoint(new THREE.Vector3(...data.position), true);
            return;
        }
    }

    // 2. Check standard tactical targets
    const hits = raycaster.intersectObjects(tacticalTargetGroup.children, true);
    if (hits.length > 0) {
        let root = hits[0].object;
        while (root.parent && root.parent !== tacticalTargetGroup) {
            root = root.parent;
        }
        if (root.userData && root.userData.id) {
            const data = root.userData;
            const measureBox = document.getElementById("measure-readout");
            if (measureBox) {
                measureBox.innerHTML = `
                    ID: <b>${data.id}</b><br>
                    TAG: <b>${data.label}</b><br>
                    THREAT: <b style="color:${data.threat_level === 'HIGH' ? '#ef4444' : '#3b82f6'}">${data.threat_level}</b>
                `;
            }
            focusOnPoint(root.children[0].position, true);
            return;
        }
    }

    // 3. Check BRO road potholes
    if (isRoadPotholesActive && roadPotholeGroup && roadPotholeGroup.visible) {
        const potHits = raycaster.intersectObjects(roadPotholeGroup.children, true);
        if (potHits.length > 0) {
            let root = potHits[0].object;
            while (root.parent && root.parent !== roadPotholeGroup) {
                root = root.parent;
            }
            if (root.userData && root.userData.id) {
                selectPothole(root.userData);
                return;
            }
        }
    }
});

// =================================================================
// 9. RESPONSIVE RESIZE & ANIMATION LOOP
// =================================================================
window.addEventListener("resize", () => {
    if (!container.clientWidth || !container.clientHeight) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Pulse animation for critical potholes if active
    if (isRoadPotholesActive && roadPotholeGroup && roadPotholeGroup.visible) {
        const time = Date.now() * 0.003;
        roadPotholeMarkers.forEach(m => {
            if (m.userData && m.userData.severity === "CRITICAL" && m.pulseRing) {
                const s = 1.0 + Math.sin(time * 2.5) * 0.16;
                m.pulseRing.scale.set(s, s, 1);
            }
        });
    }

    renderer.render(scene, camera);
}
animate();

// =================================================================
// 10. TACTICAL INGESTION & PIPELINE CONTROLLER
// =================================================================
let activeInputType = "video";   // "video" | "frames"
let activeHasTelemetry = true;    // true | false
let activeQuality = "medium";    // "fast" | "medium" | "high"
let selectedMediaFile = null;
let selectedTelemetryFile = null;
let pipelinePollTimer = null;
let pipelineStartTime = null;

// DOM Elements
const ingestModal = document.getElementById("ingest-modal");
const openIngestBtn = document.getElementById("open-ingest-btn");
const closeModalBtn = document.getElementById("close-modal-btn");

const modeVideoBtn = document.getElementById("mode-video-btn");
const modeFramesBtn = document.getElementById("mode-frames-btn");
const modeColabBtn = document.getElementById("mode-colab-btn");
const coordsYesBtn = document.getElementById("coords-yes-btn");
const coordsNoBtn = document.getElementById("coords-no-btn");

const qualityFastBtn = document.getElementById("quality-fast-btn");
const qualityMedBtn = document.getElementById("quality-med-btn");
const qualityHighBtn = document.getElementById("quality-high-btn");
const qualityUltraBtn = document.getElementById("quality-ultra-btn");

const qualityBtns = [qualityFastBtn, qualityMedBtn, qualityHighBtn, qualityUltraBtn];
qualityBtns.forEach(btn => {
    if (!btn) return;
    btn.addEventListener("click", () => {
        qualityBtns.forEach(b => b && b.classList.remove("active"));
        btn.classList.add("active");
        activeQuality = btn.getAttribute("data-quality");

        const hintEl = document.getElementById("quality-mode-hint");
        if (activeQuality === "fast") hintEl.textContent = "~3-5 min (Fast Tactical Demo)";
        else if (activeQuality === "medium") hintEl.textContent = "~12-15 min (NTRO Spec Standard)";
        else if (activeQuality === "high") hintEl.textContent = "~20-25 min (High Detail MVS Cloud)";
        else if (activeQuality === "ultra") hintEl.textContent = "~25-30 min (Ultra Poisson Mesh & Blender .OBJ)";
    });
});

const mediaDropzone = document.getElementById("media-dropzone");
const mediaFileInput = document.getElementById("media-file-input");
const mediaDropzoneTitle = document.getElementById("media-dropzone-title");
const mediaDropzoneSub = document.getElementById("media-dropzone-sub");
const mediaFileBadge = document.getElementById("media-file-badge");
const mediaFilename = document.getElementById("media-filename");
const mediaFilesize = document.getElementById("media-filesize");

const telemetryDropzone = document.getElementById("telemetry-dropzone");
const telemetryFileInput = document.getElementById("telemetry-file-input");
const telemetryFileBadge = document.getElementById("telemetry-file-badge");
const telemetryFilename = document.getElementById("telemetry-filename");
const telemetryFilesize = document.getElementById("telemetry-filesize");

const startPipelineBtn = document.getElementById("start-pipeline-btn");
const cancelPipelineBtn = document.getElementById("cancel-pipeline-btn");

const pipelineMonitor = document.getElementById("pipeline-monitor");
const monitorStageText = document.getElementById("monitor-stage-text");
const monitorTimer = document.getElementById("monitor-timer");
const progressBarFill = document.getElementById("progress-bar-fill");
const pipelineTerminal = document.getElementById("pipeline-terminal");

const pipelineStatusBadge = document.getElementById("pipeline-status-badge");
const pipelineDot = document.getElementById("pipeline-dot");
const pipelineStatusText = document.getElementById("pipeline-status-text");

// Modal open/close
if (openIngestBtn && ingestModal) {
    openIngestBtn.addEventListener("click", () => {
        ingestModal.classList.add("active");
    });
}

if (closeModalBtn && ingestModal) {
    closeModalBtn.addEventListener("click", () => {
        ingestModal.classList.remove("active");
    });
}

// Input Type Toggle
function updateInputModeUI() {
    if (activeInputType === "video") {
        if (modeVideoBtn) modeVideoBtn.classList.add("active");
        if (modeFramesBtn) modeFramesBtn.classList.remove("active");
        if (modeColabBtn) modeColabBtn.classList.remove("active");
        document.getElementById("input-mode-hint").textContent = "Drone Video Stream";
        mediaDropzoneTitle.textContent = "DRAG & DROP DRONE VIDEO (.MP4 / .MOV)";
        mediaDropzoneSub.textContent = "or click to select single-pass flyover video";
        mediaFileInput.accept = ".mp4,.mov,.avi,.mkv";
        if (startPipelineBtn) startPipelineBtn.textContent = "INITIATE 3D RECONSTRUCTION";
        if (telemetryDropzone) telemetryDropzone.style.display = "flex";
    } else if (activeInputType === "frames") {
        if (modeFramesBtn) modeFramesBtn.classList.add("active");
        if (modeVideoBtn) modeVideoBtn.classList.remove("active");
        if (modeColabBtn) modeColabBtn.classList.remove("active");
        document.getElementById("input-mode-hint").textContent = "Extracted Frames Archive";
        mediaDropzoneTitle.textContent = "DRAG & DROP FRAMES ARCHIVE (.ZIP)";
        mediaDropzoneSub.textContent = "ZIP of image frames (raw frame dataset)";
        mediaFileInput.accept = ".zip";
        if (startPipelineBtn) startPipelineBtn.textContent = "INITIATE 3D RECONSTRUCTION";
        if (telemetryDropzone) telemetryDropzone.style.display = "flex";
    } else if (activeInputType === "colab") {
        if (modeColabBtn) modeColabBtn.classList.add("active");
        if (modeVideoBtn) modeVideoBtn.classList.remove("active");
        if (modeFramesBtn) modeFramesBtn.classList.remove("active");
        document.getElementById("input-mode-hint").textContent = "Cloud Package (Google Colab / Kaggle)";
        mediaDropzoneTitle.textContent = "DRAG & DROP COLAB BUNDLE (.ZIP)";
        mediaDropzoneSub.textContent = "Select prism_colab_bundle.zip (bypasses local GPU entirely)";
        mediaFileInput.accept = ".zip";
        if (startPipelineBtn) startPipelineBtn.textContent = "DEPLOY CLOUD MISSION";
        if (telemetryDropzone) telemetryDropzone.style.display = "none";
    }
}

if (modeVideoBtn) {
    modeVideoBtn.addEventListener("click", () => {
        activeInputType = "video";
        updateInputModeUI();
    });
}

if (modeFramesBtn) {
    modeFramesBtn.addEventListener("click", () => {
        activeInputType = "frames";
        updateInputModeUI();
    });
}

if (modeColabBtn) {
    modeColabBtn.addEventListener("click", () => {
        activeInputType = "colab";
        updateInputModeUI();
    });
}

// Coordinates Toggle
if (coordsYesBtn && coordsNoBtn) {
    coordsYesBtn.addEventListener("click", () => {
        activeHasTelemetry = true;
        coordsYesBtn.classList.add("active");
        coordsNoBtn.classList.remove("active");
        document.getElementById("coords-mode-hint").textContent = "Attached Coordinates (DJI/Mid-Air)";
        telemetryDropzone.style.display = "block";
    });

    coordsNoBtn.addEventListener("click", () => {
        activeHasTelemetry = false;
        coordsNoBtn.classList.add("active");
        coordsYesBtn.classList.remove("active");
        document.getElementById("coords-mode-hint").textContent = "No Coordinates (Pure SfM Fallback)";
        telemetryDropzone.style.display = "none";
        selectedTelemetryFile = null;
        if (telemetryFileBadge) telemetryFileBadge.classList.remove("visible");
    });
}

// Dropzone 1: Media
if (mediaDropzone && mediaFileInput) {
    mediaDropzone.addEventListener("click", () => mediaFileInput.click());

    mediaDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        mediaDropzone.classList.add("dragover");
    });
    mediaDropzone.addEventListener("dragleave", () => mediaDropzone.classList.remove("dragover"));
    mediaDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        mediaDropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleMediaSelection(e.dataTransfer.files[0]);
        }
    });

    mediaFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleMediaSelection(e.target.files[0]);
        }
    });
}

function handleMediaSelection(file) {
    selectedMediaFile = file;
    mediaFilename.textContent = file.name;
    const mb = (file.size / (1024 * 1024)).toFixed(2);
    mediaFilesize.textContent = `${mb} MB`;
    mediaFileBadge.classList.add("visible");

    // Auto-detect Colab package
    const lower = file.name.toLowerCase();
    if (lower.endsWith(".zip") && (lower.includes("colab") || lower.includes("bundle") || lower.includes("prism"))) {
        activeInputType = "colab";
        updateInputModeUI();
        showToast(`Auto-detected Cloud Package: ${file.name}`);
    } else {
        showToast(`Media selected: ${file.name}`);
    }
}

// Dropzone 2: Telemetry
if (telemetryDropzone && telemetryFileInput) {
    telemetryDropzone.addEventListener("click", () => telemetryFileInput.click());

    telemetryDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        telemetryDropzone.classList.add("dragover");
    });
    telemetryDropzone.addEventListener("dragleave", () => telemetryDropzone.classList.remove("dragover"));
    telemetryDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        telemetryDropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleTelemetrySelection(e.dataTransfer.files[0]);
        }
    });

    telemetryFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleTelemetrySelection(e.target.files[0]);
        }
    });
}

function handleTelemetrySelection(file) {
    selectedTelemetryFile = file;
    telemetryFilename.textContent = file.name;
    const kb = (file.size / 1024).toFixed(1);
    telemetryFilesize.textContent = `${kb} KB`;
    telemetryFileBadge.classList.add("visible");
    showToast(`Telemetry selected: ${file.name}`);
}

// Start Reconstruction Pipeline
if (startPipelineBtn) {
    startPipelineBtn.addEventListener("click", async () => {
        if (!selectedMediaFile) {
            alert(activeInputType === "colab"
                ? "Please select the Google Colab package (.zip) to deploy!"
                : "Please select a Drone Video (.mp4) or Frames archive (.zip) to begin!");
            return;
        }

        const isColabZip = selectedMediaFile && selectedMediaFile.name.toLowerCase().endsWith(".zip") &&
            (activeInputType === "colab" || selectedMediaFile.name.toLowerCase().includes("colab") || selectedMediaFile.name.toLowerCase().includes("bundle") || selectedMediaFile.name.toLowerCase().includes("prism"));

        // Direct Colab Cloud Bundle Ingestion (Zero local GPU stress)
        if (activeInputType === "colab" || isColabZip) {
            const formData = new FormData();
            formData.append("bundle_file", selectedMediaFile);

            startPipelineBtn.disabled = true;
            startPipelineBtn.style.opacity = "0.5";
            pipelineMonitor.classList.add("active");
            pipelineTerminal.textContent = ">>> Ingesting Google Colab Cloud Mission Bundle...\n>>> Deploying 3D Point Cloud, Blender Mesh, Telemetry and Video...\n";
            progressBarFill.style.width = "45%";
            monitorStageText.textContent = "DEPLOYING CLOUD MISSION ASSETS...";

            try {
                const res = await fetch(`${API_BASE}/api/pipeline/import-colab`, {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Failed to deploy Colab mission bundle");

                progressBarFill.style.width = "100%";
                monitorStageText.textContent = "CLOUD MISSION DEPLOYED!";
                pipelineTerminal.textContent += `>>> ${data.message}\n>>> Assets: ${data.deployed_files.join(", ")}\n>>> Points: ${data.vertex_count.toLocaleString()} | Mesh Faces: ${data.face_count.toLocaleString()}\n`;
                showToast("SUCCESS! Cloud Mission Bundle deployed.");

                setTimeout(() => {
                    if (ingestModal) ingestModal.classList.remove("active");
                    startPipelineBtn.disabled = false;
                    startPipelineBtn.style.opacity = "1";
                    pipelineMonitor.classList.remove("active");
                    loadActiveModel(() => {
                        loadTacticalTargets();
                    });
                    const uavVideo = document.getElementById("uav-video");
                    if (uavVideo) {
                        uavVideo.src = `${API_BASE}/data/raw_videos/drone_flight.mp4?t=${Date.now()}`;
                        uavVideo.load();
                    }
                    if (typeof fetchTelemetry === "function") fetchTelemetry();
                }, 1200);
            } catch (err) {
                alert(`Cloud Bundle Ingestion Error: ${err.message}`);
                startPipelineBtn.disabled = false;
                startPipelineBtn.style.opacity = "1";
            }
            return;
        }

        if (activeHasTelemetry && !selectedTelemetryFile) {
            const proceedWithout = confirm("No telemetry file was uploaded. Proceed using automated coordinate estimation?");
            if (!proceedWithout) return;
            activeHasTelemetry = false;
        }

        const formData = new FormData();
        formData.append("input_type", activeInputType);
        formData.append("has_telemetry", activeHasTelemetry ? "true" : "false");
        formData.append("quality", activeQuality);
        formData.append("media_file", selectedMediaFile);
        if (selectedTelemetryFile) {
            formData.append("telemetry_file", selectedTelemetryFile);
        }

        try {
            startPipelineBtn.disabled = true;
            startPipelineBtn.style.opacity = "0.5";
            if (cancelPipelineBtn) cancelPipelineBtn.style.display = "inline-flex";

            pipelineMonitor.classList.add("active");
            pipelineTerminal.textContent = ">>> Ingestion payload submitted to PRISM Backend Engine...\n";
            progressBarFill.style.width = "4%";
            monitorStageText.textContent = "INITIALIZING BACKGROUND RECONSTRUCTION...";

            pipelineStatusBadge.className = "status-badge running";
            pipelineDot.className = "dot running";
            pipelineStatusText.textContent = "PIPELINE: ACTIVE (0%)";

            pipelineStartTime = Date.now();

            const res = await fetch(`${API_BASE}/api/pipeline/start`, {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || "Failed to initiate pipeline");
            }

            const data = await res.json();
            showToast(`Pipeline Initiated: ${data.job_id}`);

            // Start polling status
            startStatusPolling();
        } catch (err) {
            alert(`Error: ${err.message}`);
            startPipelineBtn.disabled = false;
            startPipelineBtn.style.opacity = "1";
            if (cancelPipelineBtn) cancelPipelineBtn.style.display = "none";
        }
    });
}

// Cancel Pipeline
if (cancelPipelineBtn) {
    cancelPipelineBtn.addEventListener("click", async () => {
        const confirmCancel = confirm("Are you sure you want to abort the 3D reconstruction?");
        if (!confirmCancel) return;

        try {
            await fetch(`${API_BASE}/api/pipeline/cancel`, { method: "POST" });
            showToast("Cancellation requested.");
        } catch (e) {
            console.error(e);
        }
    });
}

function startStatusPolling() {
    if (pipelinePollTimer) clearInterval(pipelinePollTimer);

    pipelinePollTimer = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/pipeline/status?t=${Date.now()}`);
            if (!res.ok) return;

            const state = await res.json();
            updatePipelineUI(state);

            if (state.status === "completed" || state.status === "failed") {
                clearInterval(pipelinePollTimer);
                pipelinePollTimer = null;
                startPipelineBtn.disabled = false;
                startPipelineBtn.style.opacity = "1";
                if (cancelPipelineBtn) cancelPipelineBtn.style.display = "none";

                if (state.status === "completed") {
                    showToast("RECONSTRUCTION COMPLETE! Model Reloading...");
                    // Hot reload the model and telemetry without reloading page!
                    loadActiveModel(() => {
                        fetchTelemetry();
                        loadTacticalTargets();
                        // Refresh video feed
                        if (video) {
                            video.src = `${API_BASE}/data/raw_videos/drone_flight.mp4?t=${Date.now()}`;
                            video.load();
                            video.play().catch(() => {});
                        }
                    });
                }
            }
        } catch (err) {
            console.warn("Polling error:", err);
        }
    }, 1500);
}

function updatePipelineUI(state) {
    // Timer
    if (pipelineStartTime) {
        const elapsedSec = Math.floor((Date.now() - pipelineStartTime) / 1000);
        const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
        const secs = String(elapsedSec % 60).padStart(2, "0");
        monitorTimer.textContent = `T+ ${mins}:${secs}`;
    }

    // Stage text & progress bar
    monitorStageText.textContent = `${state.current_stage} (${state.progress_percent}%)`;
    progressBarFill.style.width = `${state.progress_percent}%`;

    // Top bar status
    if (state.status === "running") {
        pipelineStatusBadge.className = "status-badge running";
        pipelineDot.className = "dot running";
        pipelineStatusText.textContent = `PIPELINE: ${state.progress_percent}%`;
    } else if (state.status === "completed") {
        pipelineStatusBadge.className = "status-badge active";
        pipelineDot.className = "dot";
        pipelineStatusText.textContent = "PIPELINE: READY";
    } else if (state.status === "failed") {
        pipelineStatusBadge.className = "status-badge";
        pipelineDot.style.backgroundColor = "var(--accent-red)";
        pipelineStatusText.textContent = "PIPELINE: ERROR";
    }

    // Terminal console logs
    if (state.logs && state.logs.length > 0) {
        pipelineTerminal.textContent = state.logs.join("\n");
        pipelineTerminal.scrollTop = pipelineTerminal.scrollHeight;
    }
}

// Toast notification helper
function showToast(msg) {
    const toast = document.getElementById("tactical-toast");
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, 3200);
}

// =================================================================
// 11. MULTI-EPOCH 3D CHANGE DETECTION & BASELINE ARCHIVAL CONTROLLER
// =================================================================

let selectedBaselineId = null;
let selectedDiffThreshold = 1.5;
let currentDiffAlerts = [];
let isDiffViewActive = false;
let currentDisplayedBaseline = null;
let selectedLoadBaselineId = null;
let allSavedBaselines = [];

// DOM Elements: Load Baseline Modal & Return to Recon
const loadBaselineModal = document.getElementById("load-baseline-modal");
const closeLoadBaselineBtn = document.getElementById("close-load-baseline-btn");
const cancelLoadBaselineBtn = document.getElementById("cancel-load-baseline-btn");
const confirmDisplayBaselineBtn = document.getElementById("confirm-display-baseline-btn");
const loadBaselineListContainer = document.getElementById("load-baseline-list-container");
const loadBaselineSelectedHint = document.getElementById("load-baseline-selected-hint");
const loadBaselineSelectionName = document.getElementById("load-baseline-selection-name");
const returnReconBtn = document.getElementById("return-recon-btn");

// DOM Elements: Save Baseline Modal
const saveBaselineBtn = document.getElementById("save-baseline-btn");
const saveBaselineModal = document.getElementById("save-baseline-modal");
const closeSaveBaselineBtn = document.getElementById("close-save-baseline-btn");
const baselineNameInput = document.getElementById("baseline-name-input");
const saveBaselineVerts = document.getElementById("save-baseline-verts");
const confirmSaveBaselineBtn = document.getElementById("confirm-save-baseline-btn");

// DOM Elements: Compare Modal
const compareBaselineBtn = document.getElementById("compare-baseline-btn");
const compareModal = document.getElementById("compare-modal");
const closeCompareBtn = document.getElementById("close-compare-btn");
const baselineListContainer = document.getElementById("baseline-list-container");
const baselineSelectedHint = document.getElementById("baseline-selected-hint");
const compareDisjointWarning = document.getElementById("compare-disjoint-warning");
const disjointWarningText = document.getElementById("disjoint-warning-text");
const runCompareBtn = document.getElementById("run-compare-btn");

// DOM Elements: Diff Alert Drawer & View Switcher
const diffAlertDrawer = document.getElementById("diff-alert-drawer");
const closeAlertDrawerBtn = document.getElementById("close-alert-drawer-btn");
const alertCountPill = document.getElementById("alert-count-pill");
const alertOverlapInfo = document.getElementById("alert-overlap-info");
const alertThreshInfo = document.getElementById("alert-thresh-info");
const alertListContainer = document.getElementById("alert-list-container");

const diffViewSwitcher = document.getElementById("diff-view-switcher");
const viewModeDiffBtn = document.getElementById("view-mode-diff");
const viewModeReconBtn = document.getElementById("view-mode-recon");

// 1. Save Baseline Handlers
function openSaveBaselineModal() {
    if (!saveBaselineModal) return;
    saveBaselineModal.classList.add("active");
    if (baselineNameInput) {
        const now = new Date();
        baselineNameInput.value = `Sector Recon Baseline (${now.toLocaleDateString([], {month: 'short', day: 'numeric'})} ${now.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})})`;
    }
    if (saveBaselineVerts) {
        const tag = document.getElementById("model-tag");
        saveBaselineVerts.textContent = tag ? tag.textContent.split("|")[0].trim() : "Active Scan";
    }
}

if (saveBaselineBtn && saveBaselineModal) {
    saveBaselineBtn.addEventListener("click", openSaveBaselineModal);
}

if (closeSaveBaselineBtn && saveBaselineModal) {
    closeSaveBaselineBtn.addEventListener("click", () => {
        saveBaselineModal.classList.remove("active");
    });
}

if (confirmSaveBaselineBtn) {
    confirmSaveBaselineBtn.addEventListener("click", async () => {
        const name = baselineNameInput ? baselineNameInput.value.trim() : "";
        confirmSaveBaselineBtn.disabled = true;
        confirmSaveBaselineBtn.textContent = "Archiving...";

        try {
            const res = await fetch(`${API_BASE}/api/baseline/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: name || undefined })
            });

            const data = await res.json();
            if (res.ok && data.success) {
                showToast(`ARCHIVED: ${data.message}`);
                saveBaselineModal.classList.remove("active");
            } else {
                showToast(`Archival failed: ${data.detail || "Server error"}`);
            }
        } catch (err) {
            console.error("Save baseline error:", err);
            showToast(`Error: ${err.message}`);
        } finally {
            confirmSaveBaselineBtn.disabled = false;
            confirmSaveBaselineBtn.textContent = "ARCHIVE BASELINE";
        }
    });
}

// 2. Threshold Pill Buttons
const threshBtns = [
    document.getElementById("thresh-10-btn"),
    document.getElementById("thresh-15-btn"),
    document.getElementById("thresh-25-btn"),
    document.getElementById("thresh-35-btn")
];

threshBtns.forEach(btn => {
    if (!btn) return;
    btn.addEventListener("click", () => {
        threshBtns.forEach(b => b && b.classList.remove("active"));
        btn.classList.add("active");
        selectedDiffThreshold = parseFloat(btn.getAttribute("data-thresh")) || 1.5;
        const hint = document.getElementById("thresh-hint");
        if (hint) {
            if (selectedDiffThreshold === 1.0) hint.textContent = "≥ 1.0m (High Sensitivity: Small Barricades / Tents)";
            else if (selectedDiffThreshold === 1.5) hint.textContent = "≥ 1.5m (Standard Military: Tents / Vehicles)";
            else if (selectedDiffThreshold === 2.5) hint.textContent = "≥ 2.5m (Fortifications & Outposts)";
            else if (selectedDiffThreshold === 3.5) hint.textContent = "≥ 3.5m (Watchtowers & Radar Masts)";
        }
    });
});

// 3. Compare Modal Handlers
async function openCompareModal() {
    if (!compareModal) return;
    compareModal.classList.add("active");
    if (compareDisjointWarning) compareDisjointWarning.classList.remove("active");
    await loadBaselinesList();
}

if (compareBaselineBtn && compareModal) {
    compareBaselineBtn.addEventListener("click", openCompareModal);
}

if (closeCompareBtn && compareModal) {
    closeCompareBtn.addEventListener("click", () => {
        compareModal.classList.remove("active");
    });
}

async function loadBaselinesList() {
    if (!baselineListContainer) return;
    baselineListContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 0.75rem; padding: 10px; text-align: center;">Loading archived baselines...</div>`;

    try {
        const res = await fetch(`${API_BASE}/api/baselines?t=${Date.now()}`);
        if (!res.ok) throw new Error("Could not retrieve baselines");
        const data = await res.json();

        if (!data.baselines || data.baselines.length === 0) {
            baselineListContainer.innerHTML = `
                <div style="color: var(--accent-amber); font-size: 0.75rem; padding: 14px; text-align: center; border: 1px dashed var(--panel-border); border-radius: 4px;">
                    No archived baselines found.<br>
                    <span style="color: var(--text-muted); font-size: 0.68rem;">Click 'Save Baseline' first on any scan to establish your Day-1 ground truth.</span>
                </div>
            `;
            if (baselineSelectedHint) baselineSelectedHint.textContent = "No baselines available";
            if (runCompareBtn) runCompareBtn.disabled = true;
            return;
        }

        if (runCompareBtn) runCompareBtn.disabled = false;
        baselineListContainer.innerHTML = "";

        // Default to first baseline
        if (!selectedBaselineId || !data.baselines.some(b => b.id === selectedBaselineId)) {
            selectedBaselineId = data.baselines[0].id;
        }

        data.baselines.forEach((b, idx) => {
            const card = document.createElement("div");
            card.className = `baseline-card ${b.id === selectedBaselineId ? 'selected' : ''}`;
            card.dataset.id = b.id;

            const dateStr = b.created_at ? new Date(b.created_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : 'Saved Baseline';
            const verts = b.vertex_count ? `${b.vertex_count.toLocaleString()} pts` : `${b.size_mb || 0} MB`;
            let gpsStr = "Visual Coordinate Frame";
            if (b.telemetry_bounds) {
                gpsStr = `Lat: ${b.telemetry_bounds.center_lat.toFixed(4)}, Lon: ${b.telemetry_bounds.center_lon.toFixed(4)}`;
            }

            const meshBadge = b.has_mesh ? '<span style="font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3);">MESH READY</span>' : '';
            const srtBadge = b.has_srt ? '<span style="font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan); border: 1px solid rgba(56, 189, 248, 0.3);">SRT READY</span>' : '';
            const vidBadge = b.has_video ? '<span style="font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3);">VIDEO</span>' : '';

            card.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 3px; flex: 1;">
                    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                        <strong style="color: var(--text-main); font-size: 0.82rem;">${b.name || b.id}</strong>
                        ${meshBadge}
                        ${srtBadge}
                        ${vidBadge}
                    </div>
                    <span style="color: var(--text-muted); font-size: 0.68rem;">Date: ${dateStr} | GPS: ${gpsStr}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="text-align: right; font-size: 0.72rem; color: var(--accent-blue);">
                        <b>${verts}</b>
                    </div>
                    <button class="btn-clean-mini btn-baseline-view-direct" title="Display this baseline model on the main screen" data-id="${b.id}" type="button" style="color: var(--accent-cyan); border-color: rgba(56, 189, 248, 0.4); padding: 2px 7px; font-size: 0.68rem;">
                        View
                    </button>
                    <button class="btn-baseline-delete" title="Delete Baseline Archive" data-id="${b.id}" type="button">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            <line x1="10" y1="11" x2="10" y2="17"></line>
                            <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                    </button>
                </div>
            `;

            const viewDirectBtn = card.querySelector(".btn-baseline-view-direct");
            if (viewDirectBtn) {
                viewDirectBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    if (compareModal) compareModal.classList.remove("active");
                    displayBaselineModel(b);
                });
            }

            const delBtn = card.querySelector(".btn-baseline-delete");
            if (delBtn) {
                delBtn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const baselineName = b.name || b.id;
                    const confirmed = confirm(`Are you sure you want to permanently delete baseline:\n"${baselineName}"?\n\nThis will remove the archived 3D model and telemetry.`);
                    if (!confirmed) return;

                    delBtn.disabled = true;
                    try {
                        const res = await fetch(`${API_BASE}/api/baseline/${encodeURIComponent(b.id)}`, { method: "DELETE" });
                        const resData = await res.json();
                        if (!res.ok) throw new Error(resData.detail || "Failed to delete baseline");
                        showToast(`Baseline "${baselineName}" deleted.`);
                        if (selectedBaselineId === b.id) {
                            selectedBaselineId = null;
                        }
                        await loadBaselinesList();
                    } catch (err) {
                        console.error("Delete baseline error:", err);
                        showToast(`Error deleting baseline: ${err.message}`);
                        delBtn.disabled = false;
                    }
                });
            }

            card.addEventListener("click", () => {
                document.querySelectorAll(".baseline-card").forEach(c => c.classList.remove("selected"));
                card.classList.add("selected");
                selectedBaselineId = b.id;
                if (baselineSelectedHint) baselineSelectedHint.textContent = b.name || b.id;
                if (compareDisjointWarning) compareDisjointWarning.classList.remove("active");
            });

            baselineListContainer.appendChild(card);
        });

        const activeBaseline = data.baselines.find(b => b.id === selectedBaselineId);
        if (baselineSelectedHint && activeBaseline) {
            baselineSelectedHint.textContent = activeBaseline.name || activeBaseline.id;
        }
    } catch (err) {
        console.error("Failed loading baselines:", err);
        baselineListContainer.innerHTML = `<div style="color: var(--accent-red); font-size: 0.75rem; padding: 10px;">Failed to load baselines: ${err.message}</div>`;
    }
}

// 4. Run Comparison Execution
if (runCompareBtn) {
    runCompareBtn.addEventListener("click", async () => {
        if (!selectedBaselineId) {
            showToast("Please select a baseline to compare against!");
            return;
        }

        const originalText = runCompareBtn.textContent;
        runCompareBtn.disabled = true;
        runCompareBtn.textContent = "Analyzing Spatial Overlap & Height Deltas...";
        if (compareDisjointWarning) compareDisjointWarning.classList.remove("active");

        showToast("Initiating 3D Change Detection & Structural Height Audit...");

        try {
            const res = await fetch(`${API_BASE}/api/diff/compare`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    baseline_id: selectedBaselineId,
                    height_threshold: selectedDiffThreshold
                })
            });

            const data = await res.json();

            // Check if flights are disjoint (zero spatial overlap or coordinate mismatch)
            if (data.status === "disjoint" || data.location_match === false) {
                if (compareDisjointWarning && disjointWarningText) {
                    const sepText = data.separation_distance_m ? ` (${data.separation_distance_m >= 1000 ? (data.separation_distance_m / 1000).toFixed(2) + ' km' : data.separation_distance_m + 'm'} apart)` : '';
                    disjointWarningText.innerHTML = `
                        <strong>${data.message || 'Location Mismatch: Baseline and Recon models do not share the same coordinates.'}</strong><br>
                        <span style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; display: block;">
                            ${sepText ? 'Sortie locations are separated ' + sepText + '. ' : ''}
                            Differential risk and temporal change analysis can only be performed between flights covering the same sector.
                        </span>
                    `;
                    compareDisjointWarning.classList.add("active");
                }
                showToast("Location Mismatch: Baseline & Recon locations do not match.");
                renderAlertDrawer({
                    status: "disjoint",
                    location_match: false,
                    message: data.message,
                    alert_count: 0,
                    overlap_percent: 0,
                    alerts: []
                });
                return;
            }

            if (!res.ok) {
                throw new Error(data.detail || "Comparison calculation failed");
            }

            // Successful comparison
            compareModal.classList.remove("active");
            showToast(`COMPARISON COMPLETE! ${data.alert_count} structural changes detected (${data.overlap_percent}% coverage).`);

            // Activate view mode switcher
            if (diffViewSwitcher) diffViewSwitcher.style.display = "flex";
            if (viewModeDiffBtn && viewModeReconBtn) {
                viewModeDiffBtn.classList.add("active");
                viewModeReconBtn.classList.remove("active");
            }
            isDiffViewActive = true;

            // Load diff threat map model into viewer
            loadActiveModel(() => {
                render3DAlertMarkers(data.alerts);
                renderAlertDrawer(data);
            }, `data/models/${data.diff_model_filename}`);

        } catch (err) {
            console.error("Comparison error:", err);
            showToast(`Error running 3D diff: ${err.message}`);
        } finally {
            runCompareBtn.disabled = false;
            runCompareBtn.textContent = originalText;
        }
    });
}

// 5. Render 3D Alert Markers & Bounding Boxes
function render3DAlertMarkers(alerts) {
    // Clear previous alert objects
    diffAlertObjects.forEach(obj => diffAlertGroup.remove(obj));
    diffAlertObjects = [];
    currentDiffAlerts = alerts || [];

    if (!alerts || alerts.length === 0) return;

    alerts.forEach(alt => {
        const [w, h, d] = alt.dimensions;
        const [x, y, z] = alt.position;

        // Tactical 3D bounding box (Crisson Red)
        const boxGeo = new THREE.BoxGeometry(w, h, d);
        const edgeGeo = new THREE.EdgesGeometry(boxGeo);
        const boxMat = new THREE.LineBasicMaterial({ color: 0xdc2626, linewidth: 2 });
        const boxWire = new THREE.LineSegments(edgeGeo, boxMat);
        boxWire.position.set(x, y, z);

        // Inverted pyramid beacon pointing down at structure centroid
        const beaconGeo = new THREE.ConeGeometry(0.35, 0.8, 4);
        const beaconMat = new THREE.MeshBasicMaterial({ color: 0xdc2626, wireframe: true });
        const beacon = new THREE.Mesh(beaconGeo, beaconMat);
        beacon.rotation.x = Math.PI;
        beacon.position.set(x, y + (h / 2) + 0.65, z);

        const group = new THREE.Group();
        group.add(boxWire);
        group.add(beacon);
        group.userData = alt;

        diffAlertGroup.add(group);
        diffAlertObjects.push(group);
    });

    console.log(`Mounted ${alerts.length} 3D tactical structure alert beacons.`);
}

// 6. Populate Tactical Alert Drawer
function renderAlertDrawer(data) {
    if (!diffAlertDrawer || !alertListContainer) return;

    // Handle Disjoint / Location Mismatch explicitly
    if (data.status === "disjoint" || data.location_match === false) {
        if (alertCountPill) {
            alertCountPill.textContent = "MISMATCH";
            alertCountPill.style.background = "var(--accent-amber)";
        }
        if (alertOverlapInfo) alertOverlapInfo.textContent = "Coverage: 0% (Disjoint)";
        if (alertThreshInfo) alertThreshInfo.textContent = "Status: Location Mismatch";

        alertListContainer.innerHTML = `
            <div style="background: rgba(210, 153, 34, 0.12); border: 1px solid var(--accent-amber); border-radius: 4px; padding: 12px; color: #e3b341; font-size: 0.76rem; line-height: 1.4;">
                <div style="font-weight: 700; display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                    <span>LOCATION MISMATCH DETECTED</span>
                </div>
                <div>${data.message || "Selected baseline model and active recon scan do not cover the same geographical sector."}</div>
                <div style="margin-top: 6px; font-size: 0.68rem; color: var(--text-muted);">
                    Comparison cannot determine terrain elevations or structural risk across unrelated sites.
                </div>
            </div>
        `;
        diffAlertDrawer.classList.add("active");
        return;
    }

    if (alertCountPill) {
        alertCountPill.textContent = data.alert_count;
        alertCountPill.style.background = "var(--accent-red)";
    }
    if (alertOverlapInfo) alertOverlapInfo.textContent = `Shared Coverage: ${data.overlap_percent}%`;
    if (alertThreshInfo) alertThreshInfo.textContent = `Threshold: ≥ ${data.height_threshold_m}m`;

    alertListContainer.innerHTML = "";

    if (!data.alerts || data.alerts.length === 0) {
        alertListContainer.innerHTML = `
            <div style="color: var(--accent-green); font-size: 0.75rem; padding: 12px; text-align: center;">
                ZERO ANOMALIES DETECTED<br>
                <span style="color: var(--text-muted); font-size: 0.68rem;">Terrain elevation is consistent with Day-1 baseline.</span>
            </div>
        `;
    } else {
        data.alerts.forEach((alt, idx) => {
            const card = document.createElement("div");
            card.className = "alert-card";
            const threatClass = `threat-${alt.threat_level || 'HIGH'}`;
            const gpsInfo = (alt.gps_lat && alt.gps_lon) ? `Lat: ${alt.gps_lat.toFixed(5)}, Lon: ${alt.gps_lon.toFixed(5)}` : `Pos: [${alt.position[0]}, ${alt.position[2]}]`;

            card.innerHTML = `
                <div class="alert-card-top">
                    <span class="alert-id">${alt.id}</span>
                    <span class="alert-threat-badge ${threatClass}">${alt.threat_level}</span>
                </div>
                <div class="alert-desc">${alt.type}</div>
                <div class="alert-meta-grid">
                    <span>HEIGHT &Delta;H: <b>+${alt.max_height_gain_m}m</b></span>
                    <span>FOOTPRINT: <b>${alt.footprint_m2} m²</b></span>
                    <span style="grid-column: span 2;">POINTS: <b>${alt.point_count} pts</b> | ${gpsInfo}</span>
                </div>
            `;

            // Click alert card to fly camera to that structure
            card.addEventListener("click", () => {
                const targetPos = new THREE.Vector3(alt.position[0], alt.position[1], alt.position[2]);
                focusOnPoint(targetPos, true);

                const measureBox = document.getElementById("measure-readout");
                if (measureBox) {
                    measureBox.innerHTML = `
                        ALERT: <b style="color: #ef4444">${alt.id}</b><br>
                        TYPE: <b>${alt.type}</b><br>
                        HEIGHT &Delta;H: <b style="color: #ef4444">+${alt.max_height_gain_m}m</b><br>
                        FOOTPRINT: <b>${alt.footprint_m2} m²</b><br>
                        GPS: <b>${alt.gps_lat ? alt.gps_lat.toFixed(5) : '--'}, ${alt.gps_lon ? alt.gps_lon.toFixed(5) : '--'}</b>
                    `;
                }
            });

            alertListContainer.appendChild(card);
        });
    }

    diffAlertDrawer.classList.add("active");
}

if (closeAlertDrawerBtn && diffAlertDrawer) {
    closeAlertDrawerBtn.addEventListener("click", () => {
        diffAlertDrawer.classList.remove("active");
    });
}

// 7. Diff View Switcher (Alert Heatmap vs Recon Scan)
function setDiffViewActive(isDiff) {
    isDiffViewActive = isDiff;
    const checkDiff = document.getElementById("check-format-diff");
    const checkRecon = document.getElementById("check-format-recon");

    if (viewModeDiffBtn && viewModeReconBtn) {
        viewModeDiffBtn.classList.toggle("active", isDiff);
        viewModeReconBtn.classList.toggle("active", !isDiff);
    }
    if (checkDiff) checkDiff.textContent = isDiff ? "✓" : "";
    if (checkRecon) checkRecon.textContent = !isDiff ? "✓" : "";

    diffAlertGroup.visible = isDiff;
    if (diffAlertDrawer) {
        if (isDiff) diffAlertDrawer.classList.add("active");
        else diffAlertDrawer.classList.remove("active");
    }

    const modelPath = isDiff ? "data/models/actionable_threat_map_diff.ply" : "data/models/actionable_threat_map.ply";
    loadActiveModel(null, modelPath);
}

if (viewModeDiffBtn) viewModeDiffBtn.addEventListener("click", () => setDiffViewActive(true));
if (viewModeReconBtn) viewModeReconBtn.addEventListener("click", () => setDiffViewActive(false));

// =================================================================
// 11.5 SAVED MODEL / BASELINE VIEWER CONTROLLER
// =================================================================

function openLoadBaselineModal() {
    if (!loadBaselineModal) return;
    loadBaselineModal.classList.add("active");
    loadBaselinesForPicker();
}

function closeLoadBaselineModal() {
    if (!loadBaselineModal) return;
    loadBaselineModal.classList.remove("active");
}

if (closeLoadBaselineBtn) {
    closeLoadBaselineBtn.addEventListener("click", closeLoadBaselineModal);
}
if (cancelLoadBaselineBtn) {
    cancelLoadBaselineBtn.addEventListener("click", closeLoadBaselineModal);
}

async function loadBaselinesForPicker() {
    if (!loadBaselineListContainer) return;
    loadBaselineListContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 0.75rem; padding: 10px; text-align: center;">Loading saved baselines...</div>`;
    if (confirmDisplayBaselineBtn) confirmDisplayBaselineBtn.disabled = true;
    if (loadBaselineSelectionName) loadBaselineSelectionName.textContent = "None";

    try {
        const res = await fetch(`${API_BASE}/api/baselines?t=${Date.now()}`);
        if (!res.ok) throw new Error("Could not retrieve baselines");
        const data = await res.json();
        allSavedBaselines = data.baselines || [];

        if (allSavedBaselines.length === 0) {
            loadBaselineListContainer.innerHTML = `
                <div style="color: var(--accent-amber); font-size: 0.75rem; padding: 14px; text-align: center; border: 1px dashed var(--panel-border); border-radius: 4px;">
                    No archived baselines found in data/baselines.<br>
                    <span style="color: var(--text-muted); font-size: 0.68rem;">Click 'Save Baseline' on any scan to create an archive.</span>
                </div>
            `;
            if (loadBaselineSelectedHint) loadBaselineSelectedHint.textContent = "No baselines found";
            return;
        }

        loadBaselineListContainer.innerHTML = "";
        
        // Select either previously selected or currentDisplayedBaseline or first
        if (!selectedLoadBaselineId || !allSavedBaselines.some(b => b.id === selectedLoadBaselineId)) {
            selectedLoadBaselineId = currentDisplayedBaseline ? currentDisplayedBaseline.id : allSavedBaselines[0].id;
        }

        allSavedBaselines.forEach((b) => {
            const card = document.createElement("div");
            const isSelected = (b.id === selectedLoadBaselineId);
            const isCurrentlyActive = (currentDisplayedBaseline && currentDisplayedBaseline.id === b.id);
            card.className = `baseline-card ${isSelected ? 'selected' : ''}`;
            card.dataset.id = b.id;

            const dateStr = b.created_at ? new Date(b.created_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : 'Saved Baseline';
            const verts = b.vertex_count ? `${b.vertex_count.toLocaleString()} pts` : `${b.size_mb || 0} MB`;
            let gpsStr = "Visual Coordinate Frame";
            if (b.telemetry_bounds) {
                gpsStr = `Lat: ${b.telemetry_bounds.center_lat.toFixed(4)}, Lon: ${b.telemetry_bounds.center_lon.toFixed(4)}`;
            }

            const meshBadge = b.has_mesh ? '<span style="font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3);">MESH READY</span>' : '';
            const srtBadge = b.has_srt ? '<span style="font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; background: rgba(56, 189, 248, 0.15); color: var(--accent-cyan); border: 1px solid rgba(56, 189, 248, 0.3);">SRT READY</span>' : '';
            const vidBadge = b.has_video ? '<span style="font-size: 0.6rem; padding: 1px 4px; border-radius: 3px; background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3);">VIDEO</span>' : '';

            card.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 3px; flex: 1;">
                    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                        <strong style="color: var(--text-main); font-size: 0.82rem;">${b.name || b.id}</strong>
                        ${isCurrentlyActive ? '<span style="font-size: 0.62rem; padding: 1px 5px; border-radius: 3px; background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.4);">ACTIVE IN VIEWPORT</span>' : ''}
                        ${meshBadge}
                        ${srtBadge}
                        ${vidBadge}
                    </div>
                    <span style="color: var(--text-muted); font-size: 0.68rem;">Date: ${dateStr} | GPS: ${gpsStr}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="text-align: right; font-size: 0.72rem; color: var(--accent-blue);">
                        <b>${verts}</b>
                    </div>
                </div>
            `;

            card.addEventListener("click", () => {
                loadBaselineListContainer.querySelectorAll(".baseline-card").forEach(c => c.classList.remove("selected"));
                card.classList.add("selected");
                selectedLoadBaselineId = b.id;
                if (loadBaselineSelectionName) loadBaselineSelectionName.textContent = b.name || b.id;
                if (loadBaselineSelectedHint) loadBaselineSelectedHint.textContent = "Ready to display";
                if (confirmDisplayBaselineBtn) confirmDisplayBaselineBtn.disabled = false;
            });

            card.addEventListener("dblclick", () => {
                displayBaselineModel(b);
                closeLoadBaselineModal();
            });

            loadBaselineListContainer.appendChild(card);
        });

        const activeChoice = allSavedBaselines.find(b => b.id === selectedLoadBaselineId);
        if (activeChoice) {
            if (loadBaselineSelectionName) loadBaselineSelectionName.textContent = activeChoice.name || activeChoice.id;
            if (confirmDisplayBaselineBtn) confirmDisplayBaselineBtn.disabled = false;
        }
    } catch (err) {
        console.error("Failed loading baselines picker:", err);
        loadBaselineListContainer.innerHTML = `<div style="color: var(--accent-red); font-size: 0.75rem; padding: 10px;">Failed to load baselines: ${err.message}</div>`;
    }
}

if (confirmDisplayBaselineBtn) {
    confirmDisplayBaselineBtn.addEventListener("click", () => {
        if (!selectedLoadBaselineId) {
            showToast("Please choose a saved baseline first.");
            return;
        }
        const b = allSavedBaselines.find(item => item.id === selectedLoadBaselineId);
        if (b) {
            displayBaselineModel(b);
            closeLoadBaselineModal();
        }
    });
}

function displayBaselineModel(baseline) {
    if (!baseline || !baseline.id) return;
    currentDisplayedBaseline = baseline;

    // Reset diff state if active
    if (isDiffViewActive) {
        setDiffViewActive(false);
    }

    const baselinePath = `data/baselines/${baseline.id}/model.ply`;
    console.log(`Displaying saved baseline model: ${baseline.name || baseline.id} (${baselinePath})`);

    // 1. Load baseline point cloud
    loadActiveModel(() => {
        updateModelTag();
    }, baselinePath);

    // 2. Seamlessly mount baseline solid mesh if pre-computed
    fetch(`${API_BASE}/data/baselines/${baseline.id}/mesh.ply?t=${Date.now()}`, { method: "HEAD" })
        .then(hRes => {
            if (hRes.ok) {
                loader.load(
                    `${API_BASE}/data/baselines/${baseline.id}/mesh.ply?t=${Date.now()}`,
                    (meshGeo) => {
                        meshGeo.translate(-savedModelCenter.x, -savedModelCenter.y, -savedModelCenter.z);
                        meshGeo.computeVertexNormals();
                        if (solidMesh) {
                            tacticalModelGroup.remove(solidMesh);
                            if (solidMesh.geometry) solidMesh.geometry.dispose();
                        }
                        solidMesh = new THREE.Mesh(meshGeo, standardMaterial);
                        solidMesh.visible = (activeShadingMode !== "points");
                        tacticalModelGroup.add(solidMesh);
                        modelHasFaces = true;
                        updateModelTag();
                    }
                );
            }
        })
        .catch(() => {});

    // 3. Synchronize Video HUD or show Standby Overlay
    const videoEl = document.getElementById("uav-video");
    const videoStandby = document.getElementById("video-standby-overlay");
    const camWatermark = document.getElementById("cam-watermark");
    const videoInfoTag = document.getElementById("video-info-tag");

    if (baseline.has_video && baseline.video_url) {
        if (videoStandby) videoStandby.style.display = "none";
        if (videoEl) {
            videoEl.src = `${API_BASE}/${baseline.video_url}?t=${Date.now()}`;
            videoEl.load();
            videoEl.play().catch(() => {});
        }
        if (camWatermark) camWatermark.textContent = `BASELINE SORTIE ● ${(baseline.name || baseline.id).toUpperCase().substring(0, 18)}`;
        if (videoInfoTag) videoInfoTag.textContent = "ARCHIVED SORTIE";
    } else {
        if (videoStandby) videoStandby.style.display = "flex";
        if (videoEl) videoEl.pause();
        if (camWatermark) camWatermark.textContent = "BASELINE ARCHIVE ● NO VIDEO RECORDED";
        if (videoInfoTag) videoInfoTag.textContent = "STANDBY";
    }

    // 4. Wire Telemetry & SRT Intel Button
    const btnDownloadSrt = document.getElementById("btn-download-srt");
    if (btnDownloadSrt) {
        btnDownloadSrt.style.display = "inline-flex";
        btnDownloadSrt.title = `Download flight telemetry SRT for ${baseline.name || baseline.id}`;
        btnDownloadSrt.onclick = () => {
            window.open(`${API_BASE}/api/baseline/${encodeURIComponent(baseline.id)}/srt`, "_blank");
        };
    }

    // Fetch baseline telemetry to synchronize HUD indicators
    fetch(`${API_BASE}/data/baselines/${baseline.id}/telemetry.json?t=${Date.now()}`)
        .then(res => res.ok ? res.json() : null)
        .then(telem => {
            if (telem) {
                flightData = telem;
                if (telem.waypoints && telem.waypoints.length > 0) {
                    const wp = telem.waypoints[0];
                    if (latEl) latEl.textContent = Number(wp.latitude).toFixed(6);
                    if (lonEl) lonEl.textContent = Number(wp.longitude).toFixed(6);
                    if (altEl) altEl.textContent = Number(wp.relative_altitude_m).toFixed(2);
                    if (distEl) distEl.textContent = telem.total_distance_m ? telem.total_distance_m.toFixed(1) : "--";
                } else if (telem.center_lat && telem.center_lon) {
                    if (latEl) latEl.textContent = Number(telem.center_lat).toFixed(6);
                    if (lonEl) lonEl.textContent = Number(telem.center_lon).toFixed(6);
                    if (altEl) altEl.textContent = telem.avg_altitude_m ? Number(telem.avg_altitude_m).toFixed(2) : "--";
                    if (distEl) distEl.textContent = telem.total_distance_m ? telem.total_distance_m.toFixed(1) : "--";
                }
                if (telemSourceTag) {
                    const cleanName = (baseline.name || baseline.id).toUpperCase().replace(/[^A-Z0-9_\-\s]/g, "");
                    telemSourceTag.textContent = `BASELINE: ${cleanName.substring(0, 14)}`;
                }
            }
        })
        .catch(err => {
            console.warn("Could not load baseline telemetry:", err);
        });

    // 5. Show "Return to Active Recon" UI controls
    if (returnReconBtn) {
        returnReconBtn.style.display = "inline-flex";
    }
    const menuRestoreActive = document.getElementById("menu-file-restore-active");
    if (menuRestoreActive) {
        menuRestoreActive.style.display = "flex";
    }

    showToast(`VIEWING BASELINE: ${baseline.name || baseline.id}`);
}

function restoreActiveReconModel() {
    currentDisplayedBaseline = null;

    // 1. Restore Active Video HUD
    const videoEl = document.getElementById("uav-video");
    const videoStandby = document.getElementById("video-standby-overlay");
    const camWatermark = document.getElementById("cam-watermark");
    const videoInfoTag = document.getElementById("video-info-tag");

    if (videoStandby) videoStandby.style.display = "none";
    if (videoEl) {
        videoEl.src = `${API_BASE}/data/raw_videos/drone_flight.mp4`;
        videoEl.load();
        videoEl.play().catch(() => {});
    }
    if (camWatermark) camWatermark.textContent = "REC ● ACTIVE SENSOR";
    if (videoInfoTag) videoInfoTag.textContent = "1080p // 30 FPS";

    // 2. Restore Active SRT Download Action
    const btnDownloadSrt = document.getElementById("btn-download-srt");
    if (btnDownloadSrt) {
        btnDownloadSrt.title = "Download active mission telemetry SRT";
        btnDownloadSrt.onclick = () => {
            window.open(`${API_BASE}/data/drone_flight.srt`, "_blank");
        };
    }

    // 3. Hide Return to Recon UI controls
    if (returnReconBtn) {
        returnReconBtn.style.display = "none";
    }
    const menuRestoreActive = document.getElementById("menu-file-restore-active");
    if (menuRestoreActive) {
        menuRestoreActive.style.display = "none";
    }

    // Reset diff state if active
    if (isDiffViewActive) {
        setDiffViewActive(false);
    }

    // 4. Reload active reconnaissance model
    loadActiveModel(() => {
        updateModelTag();
    }, "data/models/actionable_threat_map.ply");

    // 5. Re-fetch live/active mission telemetry
    fetchTelemetry();

    showToast("RESTORED: Active mission reconstruction model mounted.");
}

// Wire Return to Recon button
if (returnReconBtn) {
    returnReconBtn.addEventListener("click", () => {
        restoreActiveReconModel();
    });
}

// =================================================================
// 12. DESKTOP APPLICATION MENU BAR & SHORTCUTS (ANTIGRAVITY IDE STYLE)
// =================================================================
const menuBtnFile = document.getElementById("menu-btn-file");
const dropdownFile = document.getElementById("dropdown-file");
const menuBtnFormat = document.getElementById("menu-btn-format");
const dropdownFormat = document.getElementById("dropdown-format");
const menuBtnTools = document.getElementById("menu-btn-tools");
const dropdownTools = document.getElementById("dropdown-tools");

const allDropdowns = [dropdownFile, dropdownFormat, dropdownTools];
const allMenuBtns = [menuBtnFile, menuBtnFormat, menuBtnTools];

function closeAllDropdowns() {
    allDropdowns.forEach(d => d && d.classList.remove("show"));
    allMenuBtns.forEach(b => b && b.classList.remove("active"));
}

function toggleMenu(btn, menu) {
    if (!btn || !menu) return;
    const wasOpen = menu.classList.contains("show");
    closeAllDropdowns();
    if (!wasOpen) {
        menu.classList.add("show");
        btn.classList.add("active");
    }
}

if (menuBtnFile) menuBtnFile.addEventListener("click", (e) => { e.stopPropagation(); toggleMenu(menuBtnFile, dropdownFile); });
if (menuBtnFormat) menuBtnFormat.addEventListener("click", (e) => { e.stopPropagation(); toggleMenu(menuBtnFormat, dropdownFormat); });
if (menuBtnTools) menuBtnTools.addEventListener("click", (e) => { e.stopPropagation(); toggleMenu(menuBtnTools, dropdownTools); });

// Close dropdowns on outside click
document.addEventListener("click", (e) => {
    if (!e.target.closest(".menu-item-wrapper")) {
        closeAllDropdowns();
    }
});

// File Menu Actions
const menuFileIngest = document.getElementById("menu-file-ingest");
if (menuFileIngest) {
    menuFileIngest.addEventListener("click", () => {
        closeAllDropdowns();
        if (ingestModal) ingestModal.classList.add("active");
    });
}

const menuFileSaveBaseline = document.getElementById("menu-file-save-baseline");
if (menuFileSaveBaseline) {
    menuFileSaveBaseline.addEventListener("click", () => {
        closeAllDropdowns();
        openSaveBaselineModal();
    });
}

const menuFileLoadBaseline = document.getElementById("menu-file-load-baseline");
if (menuFileLoadBaseline) {
    menuFileLoadBaseline.addEventListener("click", () => {
        closeAllDropdowns();
        openLoadBaselineModal();
    });
}

const menuFileRestoreActive = document.getElementById("menu-file-restore-active");
if (menuFileRestoreActive) {
    menuFileRestoreActive.addEventListener("click", () => {
        closeAllDropdowns();
        restoreActiveReconModel();
    });
}

const menuFileCompare = document.getElementById("menu-file-compare");
if (menuFileCompare) {
    menuFileCompare.addEventListener("click", () => {
        closeAllDropdowns();
        openCompareModal();
    });
}

const menuFileExportObj = document.getElementById("menu-file-export-obj");
if (menuFileExportObj) {
    menuFileExportObj.addEventListener("click", () => {
        closeAllDropdowns();
        triggerExportOBJ();
    });
}

const menuFileMeshify = document.getElementById("menu-file-meshify");
if (menuFileMeshify) {
    menuFileMeshify.addEventListener("click", () => {
        closeAllDropdowns();
        triggerMeshify();
    });
}

// Format Menu Actions
const menuFormatPoints = document.getElementById("menu-format-points");
if (menuFormatPoints) {
    menuFormatPoints.addEventListener("click", () => {
        closeAllDropdowns();
        applyShadingMode("points");
    });
}

const menuFormatMesh = document.getElementById("menu-format-mesh");
if (menuFormatMesh) {
    menuFormatMesh.addEventListener("click", () => {
        closeAllDropdowns();
        applyShadingMode("mesh");
    });
}

const menuFormatWireframe = document.getElementById("menu-format-wireframe");
if (menuFormatWireframe) {
    menuFormatWireframe.addEventListener("click", () => {
        closeAllDropdowns();
        applyShadingMode("wireframe");
    });
}

const menuFormatClay = document.getElementById("menu-format-clay");
if (menuFormatClay) {
    menuFormatClay.addEventListener("click", () => {
        closeAllDropdowns();
        applyShadingMode("clay");
    });
}

const menuFormatRecon = document.getElementById("menu-format-recon");
if (menuFormatRecon) {
    menuFormatRecon.addEventListener("click", () => {
        closeAllDropdowns();
        setDiffViewActive(false);
    });
}

const menuFormatDiff = document.getElementById("menu-format-diff");
if (menuFormatDiff) {
    menuFormatDiff.addEventListener("click", () => {
        closeAllDropdowns();
        setDiffViewActive(true);
    });
}

// Tools Menu Actions
const menuToolCaliper = document.getElementById("menu-tool-caliper");
if (menuToolCaliper) {
    menuToolCaliper.addEventListener("click", () => {
        closeAllDropdowns();
        toggleCaliperMode();
    });
}

const menuToolClearCaliper = document.getElementById("menu-tool-clear-caliper");
if (menuToolClearCaliper) {
    menuToolClearCaliper.addEventListener("click", () => {
        closeAllDropdowns();
        clearMeasurementMarkers();
        showToast("Measurement markers cleared.");
    });
}

const menuToolHeightRestriction = document.getElementById("menu-tool-height-restriction");
if (menuToolHeightRestriction) {
    menuToolHeightRestriction.addEventListener("click", () => {
        closeAllDropdowns();
        openHeightRestrictionModal();
    });
}

const menuToolClearHeight = document.getElementById("menu-tool-clear-height");
if (menuToolClearHeight) {
    menuToolClearHeight.addEventListener("click", () => {
        closeAllDropdowns();
        clearHeightRestriction();
    });
}

const menuToolPotholes = document.getElementById("menu-tool-potholes");
if (menuToolPotholes) {
    menuToolPotholes.addEventListener("click", () => {
        closeAllDropdowns();
        toggleRoadPotholesFilter();
    });
}

const menuToolMeshify = document.getElementById("menu-tool-meshify");
if (menuToolMeshify) {
    menuToolMeshify.addEventListener("click", () => {
        closeAllDropdowns();
        triggerMeshify();
    });
}

const menuToolSaveBaseline = document.getElementById("menu-tool-save-baseline");
if (menuToolSaveBaseline) {
    menuToolSaveBaseline.addEventListener("click", () => {
        closeAllDropdowns();
        openSaveBaselineModal();
    });
}

const menuToolCompare = document.getElementById("menu-tool-compare");
if (menuToolCompare) {
    menuToolCompare.addEventListener("click", () => {
        closeAllDropdowns();
        openCompareModal();
    });
}

const menuToolRecenter = document.getElementById("menu-tool-recenter");
if (menuToolRecenter) {
    menuToolRecenter.addEventListener("click", () => {
        closeAllDropdowns();
        resetCamera();
        showToast("Camera recentered to scene.");
    });
}

// =================================================================
// 13. BRO ROAD & POTHOLE 3D DEPTH ENGINE CONTROLLER
// =================================================================
const potholeFilterBtn = document.getElementById("pothole-filter-btn");
const menuFormatPotholes = document.getElementById("menu-format-potholes");
const checkFormatPotholes = document.getElementById("check-format-potholes");

const potholeAuditDrawer = document.getElementById("pothole-audit-drawer");
const closePotholeDrawerBtn = document.getElementById("close-pothole-drawer-btn");
const potholeCountPill = document.getElementById("pothole-count-pill");
const roadTypePill = document.getElementById("road-type-pill");
const pqiScoreBadge = document.getElementById("pqi-score-badge");
const roadConditionDesc = document.getElementById("road-condition-desc");
const potholeMaxDepth = document.getElementById("pothole-max-depth");
const potholeAvgDepth = document.getElementById("pothole-avg-depth");
const potholeDamagedArea = document.getElementById("pothole-damaged-area");
const potholeListContainer = document.getElementById("pothole-list-container");

const potholeCompileView = document.getElementById("pothole-compile-view");
const potholeResultsView = document.getElementById("pothole-results-view");
const potholeRunCompileBtn = document.getElementById("pothole-run-compile-btn");
const potholeCompileSpinner = document.getElementById("pothole-compile-spinner");
const potholeRecompileBtn = document.getElementById("pothole-recompile-btn");

const videoOverlayCanvas = document.getElementById("video-overlay-canvas");
const toggleVideoOverlayBtn = document.getElementById("toggle-video-overlay-btn");
const videoRoadBadge = document.getElementById("video-road-badge");
const videoRoadBadgeText = document.getElementById("video-road-badge-text");
const videoRoadDot = document.getElementById("video-road-dot");

function createTacticalDepthSprite(id, depthCm, severity, hexColor) {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 96;
    const ctx = canvas.getContext("2d");

    // Background badge
    ctx.fillStyle = "rgba(13, 17, 23, 0.92)";
    ctx.strokeStyle = `#${hexColor.toString(16).padStart(6, '0')}`;
    ctx.lineWidth = 4;

    const r = 10;
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(256 - r, 0);
    ctx.quadraticCurveTo(256, 0, 256, r);
    ctx.lineTo(256, 96 - r);
    ctx.quadraticCurveTo(256, 96, 256 - r, 96);
    ctx.lineTo(r, 96);
    ctx.quadraticCurveTo(0, 96, 0, 96 - r);
    ctx.lineTo(0, r);
    ctx.quadraticCurveTo(0, 0, r, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Top text: ID & Severity
    ctx.fillStyle = `#${hexColor.toString(16).padStart(6, '0')}`;
    ctx.font = "bold 20px monospace";
    ctx.fillText(`${id} [${severity}]`, 16, 32);

    // Hero depth readout
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 34px sans-serif";
    ctx.fillText(`-${depthCm} cm`, 16, 74);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(1.4, 0.52, 1);
    return sprite;
}

function render3DPotholeMarkers(potholes, rawCenter) {
    if (!roadPotholeGroup) return;

    // Clear previous pothole objects
    while (roadPotholeGroup.children.length > 0) {
        const obj = roadPotholeGroup.children[0];
        roadPotholeGroup.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
            else obj.material.dispose();
        }
    }
    roadPotholeMarkers = [];

    if (!potholes || potholes.length === 0) return;

    potholes.forEach((pot) => {
        const markerGroup = new THREE.Group();
        markerGroup.userData = pot;

        // Position: use centered_position in model coordinate space
        const [px, py, pz] = pot.centered_position;
        markerGroup.position.set(px, py, pz);

        const radius = Math.min(1.2, Math.max(0.35, (pot.diameter_cm / 200.0)));
        const depthM = Math.max(0.08, pot.depth_cm / 100.0);

        let sevColor = 0x60a5fa; // MINOR: cyan
        if (pot.severity === "CRITICAL") sevColor = 0xef4444; // CRITICAL: red
        else if (pot.severity === "MODERATE") sevColor = 0xf59e0b; // MODERATE: amber

        // 1. Pothole Ground Ring Decal
        const ringGeo = new THREE.RingGeometry(radius * 0.72, radius, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: sevColor,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.85
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.05;
        markerGroup.add(ring);

        // 2. Pulsing outer ring for CRITICAL hazards
        if (pot.severity === "CRITICAL") {
            const pulseGeo = new THREE.RingGeometry(radius * 1.05, radius * 1.25, 32);
            const pulseMat = new THREE.MeshBasicMaterial({
                color: sevColor,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.45
            });
            const pulseRing = new THREE.Mesh(pulseGeo, pulseMat);
            pulseRing.rotation.x = -Math.PI / 2;
            pulseRing.position.y = 0.06;
            markerGroup.add(pulseRing);
            markerGroup.pulseRing = pulseRing;
        }

        // 3. 3D Depression Cavity Cylinder (visualizing depth beneath pavement)
        const cylGeo = new THREE.CylinderGeometry(radius * 0.95, radius * 0.65, depthM, 16, 1, true);
        const cylMat = new THREE.MeshBasicMaterial({
            color: sevColor,
            wireframe: true,
            transparent: true,
            opacity: 0.65
        });
        const cyl = new THREE.Mesh(cylGeo, cylMat);
        cyl.position.y = -depthM / 2.0;
        markerGroup.add(cyl);

        // 4. Vertical Caliper Stem
        const stemHeight = 1.3;
        const stemGeo = new THREE.CylinderGeometry(0.025, 0.025, stemHeight, 8);
        const stemMat = new THREE.MeshBasicMaterial({ color: sevColor });
        const stem = new THREE.Mesh(stemGeo, stemMat);
        stem.position.y = stemHeight / 2.0;
        markerGroup.add(stem);

        // 5. Beacon Top Pin (Inverted Cone)
        const coneGeo = new THREE.ConeGeometry(0.18, 0.42, 4);
        const coneMat = new THREE.MeshBasicMaterial({ color: sevColor, wireframe: true });
        const cone = new THREE.Mesh(coneGeo, coneMat);
        cone.rotation.x = Math.PI;
        cone.position.y = stemHeight + 0.25;
        markerGroup.add(cone);

        // 6. Tactical Depth Badge Sprite
        const sprite = createTacticalDepthSprite(pot.id, pot.depth_cm, pot.severity, sevColor);
        if (sprite) {
            sprite.position.y = stemHeight + 0.75;
            markerGroup.add(sprite);
        }

        roadPotholeGroup.add(markerGroup);
        roadPotholeMarkers.push(markerGroup);
    });

    console.log(`Mounted ${potholes.length} 3D tactical road pothole markers with depths.`);
}

function selectPothole(pot) {
    if (!pot) return;
    selectedPotholeId = pot.id;

    // Highlight card in list
    document.querySelectorAll(".pothole-card").forEach(c => {
        c.classList.toggle("selected", c.dataset.id === pot.id);
    });
    const activeCard = document.querySelector(`.pothole-card[data-id="${pot.id}"]`);
    if (activeCard) {
        activeCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Convert centered local pos to world pos for camera focus
    const localPos = new THREE.Vector3(...pot.centered_position);
    const worldPos = localPos.clone().applyMatrix4(tacticalModelGroup.matrixWorld);
    focusOnPoint(worldPos, true);

    // Update measurement HUD
    const measureBox = document.getElementById("measure-readout");
    if (measureBox) {
        const color = pot.severity === "CRITICAL" ? "#ef4444" : (pot.severity === "MODERATE" ? "#f59e0b" : "#60a5fa");
        measureBox.innerHTML = `
            BRO AUDIT: <b style="color:${color}">${pot.id}</b> [${pot.severity}]<br>
            DEPTH: <b style="color:${color}">-${pot.depth_cm} cm</b> (Avg: -${pot.avg_depth_cm} cm)<br>
            DIAMETER: <b>${pot.diameter_cm} cm</b> (Area: ${pot.area_sqm} m²)<br>
            GPS: <b>${pot.gps_lat ? pot.gps_lat.toFixed(5) : '--'}, ${pot.gps_lon ? pot.gps_lon.toFixed(5) : '--'}</b><br>
            HAZARD: <span style="font-size:0.68rem; color:${color};">${pot.convoy_impact}</span>
        `;
    }

    // Seek video to corresponding flight segment
    if (video && video.duration) {
        const pNum = parseInt(pot.id.replace("POT-", "")) || 1;
        const progress = Math.min(1.0, (pNum / 8.0) * 0.95);
        video.currentTime = progress * video.duration;
    }
}

function renderPotholeDrawer(data) {
    if (!potholeAuditDrawer) return;

    if (potholeCompileView) potholeCompileView.style.display = "none";
    if (potholeResultsView) potholeResultsView.style.display = "flex";
    if (potholeRecompileBtn) potholeRecompileBtn.style.display = "inline-block";

    if (!potholeListContainer) return;

    const stats = data.statistics || {};
    const road = data.road_classification || {};

    if (potholeCountPill) {
        potholeCountPill.textContent = stats.total_potholes || 0;
        potholeCountPill.style.background = (stats.critical_potholes > 0) ? "var(--accent-red)" : "var(--accent-amber)";
    }

    if (roadTypePill) {
        const isMuddy = road.road_type === "MUDDY_UNPAVED";
        roadTypePill.textContent = isMuddy ? "MUDDY / UNPAVED (BRO TRACK)" : "TAR / ASPHALT (BRO HIGHWAY)";
        roadTypePill.classList.toggle("muddy", isMuddy);
    }

    if (pqiScoreBadge) {
        pqiScoreBadge.textContent = `PQI: ${road.pqi_score || 64}/100`;
    }

    if (roadConditionDesc) {
        roadConditionDesc.textContent = road.surface_condition || "Bituminous road pavement distress & local surface subsidence.";
    }

    if (potholeMaxDepth) potholeMaxDepth.textContent = `Max: -${stats.max_depth_cm || 0} cm`;
    if (potholeAvgDepth) potholeAvgDepth.textContent = `Avg: -${stats.average_depth_cm || 0} cm`;
    if (potholeDamagedArea) potholeDamagedArea.textContent = `Area: ${stats.total_damaged_area_sqm || 0} m²`;

    potholeListContainer.innerHTML = "";

    if (!data.potholes || data.potholes.length === 0) {
        potholeListContainer.innerHTML = `
            <div style="color: var(--accent-green); font-size: 0.75rem; padding: 14px; text-align: center;">
                ZERO POTHOLES DETECTED<br>
                <span style="color: var(--text-muted); font-size: 0.68rem;">Road corridor surface is smooth and intact.</span>
            </div>
        `;
    } else {
        data.potholes.forEach((pot) => {
            const card = document.createElement("div");
            card.className = "pothole-card";
            card.dataset.id = pot.id;

            const sevClass = `severity-${pot.severity}`;
            const heroClass = pot.severity === "CRITICAL" ? "crit" : (pot.severity === "MODERATE" ? "mod" : "min");
            const gpsInfo = (pot.gps_lat && pot.gps_lon) ? `Lat: ${pot.gps_lat.toFixed(5)}, Lon: ${pot.gps_lon.toFixed(5)}` : `Pos: [${pot.position[0]}, ${pot.position[2]}]`;

            card.innerHTML = `
                <div class="pothole-card-top">
                    <div class="pothole-id">
                        <span>${pot.id}</span>
                        <span class="pothole-depth-hero ${heroClass}">-${pot.depth_cm} cm</span>
                    </div>
                    <span class="pothole-severity-badge ${sevClass}">${pot.severity}</span>
                </div>
                <div class="alert-meta-grid">
                    <span>AVG DEPTH: <b>-${pot.avg_depth_cm} cm</b></span>
                    <span>DIAMETER: <b>${pot.diameter_cm} cm</b></span>
                    <span>AREA: <b>${pot.area_sqm} m²</b></span>
                    <span>POINTS: <b>${pot.points_in_cluster || 18} pts</b></span>
                    <span style="grid-column: span 2;">NAV: ${gpsInfo}</span>
                </div>
                <div class="pothole-convoy-tip ${heroClass}">
                    <b>HAZARD:</b> ${pot.convoy_impact}
                </div>
                <div style="font-size: 0.65rem; color: #94a3b8; display: flex; align-items: center; gap: 4px;">
                    <span style="color: var(--accent-blue);">REPAIR:</span> ${pot.recommended_action}
                </div>
            `;

            card.addEventListener("click", () => {
                selectPothole(pot);
            });

            potholeListContainer.appendChild(card);
        });
    }

    potholeAuditDrawer.classList.add("active");
}

function clearVideoOverlay() {
    if (!videoOverlayCanvas) return;
    const ctx = videoOverlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, videoOverlayCanvas.width, videoOverlayCanvas.height);
}

function updateVideoOverlay() {
    if (!videoOverlayCanvas || !video) return;
    if (!isRoadPotholesActive || !isVideoOverlayActive) {
        clearVideoOverlay();
        return;
    }

    const rect = video.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    if (videoOverlayCanvas.width !== Math.floor(rect.width) || videoOverlayCanvas.height !== Math.floor(rect.height)) {
        videoOverlayCanvas.width = Math.floor(rect.width);
        videoOverlayCanvas.height = Math.floor(rect.height);
    }

    const ctx = videoOverlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, videoOverlayCanvas.width, videoOverlayCanvas.height);

    if (!roadAuditData || !roadAuditData.video_detections || roadAuditData.video_detections.length === 0) {
        return;
    }

    const duration = video.duration || 10.0;
    const progress = Math.min(Math.max(video.currentTime / duration, 0), 1);
    const frameIdx = Math.min(
        Math.floor(progress * roadAuditData.video_detections.length),
        roadAuditData.video_detections.length - 1
    );

    const frameData = roadAuditData.video_detections[frameIdx];
    if (frameData && frameData.potholes) {
        frameData.potholes.forEach((pot) => {
            const [nx, ny, nw, nh] = pot.bbox;
            const bx = nx * videoOverlayCanvas.width;
            const by = ny * videoOverlayCanvas.height;
            const bw = nw * videoOverlayCanvas.width;
            const bh = nh * videoOverlayCanvas.height;

            const isCrit = pot.severity === "CRITICAL";
            const color = isCrit ? "#ef4444" : (pot.severity === "MODERATE" ? "#f59e0b" : "#60a5fa");

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(bx, by, bw, bh);

            ctx.fillStyle = isCrit ? "rgba(239, 68, 68, 0.16)" : "rgba(245, 158, 11, 0.12)";
            ctx.fillRect(bx, by, bw, bh);

            ctx.fillStyle = "rgba(13, 17, 23, 0.90)";
            ctx.fillRect(bx, Math.max(0, by - 18), Math.max(bw, 140), 18);
            ctx.fillStyle = color;
            ctx.font = "bold 10px monospace";
            ctx.fillText(`POTHOLE: -${pot.depth_cm}cm [${pot.severity}]`, bx + 4, Math.max(12, by - 5));
        });
    }
}

async function executeRoadAuditCompilation() {
    if (!potholeCompileSpinner || !potholeRunCompileBtn) return;

    potholeCompileSpinner.style.display = "flex";
    potholeRunCompileBtn.disabled = true;
    potholeRunCompileBtn.style.opacity = "0.6";
    if (potholeRecompileBtn) {
        potholeRecompileBtn.disabled = true;
        potholeRecompileBtn.textContent = "⏳ Computing...";
    }
    showToast("Compiling BRO Road Assessment & 3D Pothole Depths...");

    try {
        const res = await fetch(`${API_BASE}/api/road/audit/compile`, { method: "POST" });
        if (!res.ok) throw new Error("Compilation request failed");
        roadAuditData = await res.json();

        render3DPotholeMarkers(roadAuditData.potholes, roadAuditData.raw_model_center);
        renderPotholeDrawer(roadAuditData);

        if (videoRoadBadge && videoRoadBadgeText && videoRoadDot) {
            videoRoadBadge.style.display = "flex";
            const isMuddy = roadAuditData.road_classification && roadAuditData.road_classification.road_type === "MUDDY_UNPAVED";
            videoRoadBadgeText.textContent = isMuddy ? "BRO MUDDY ROAD" : "BRO TAR ROAD";
            videoRoadDot.classList.toggle("muddy", isMuddy);
        }

        updateVideoOverlay();

        const stats = roadAuditData.statistics || {};
        const road = roadAuditData.road_classification || {};
        showToast(`ROAD AUDIT COMPLETE: ${road.road_type_label || 'Corridor'} | ${stats.total_potholes || 0} Potholes (Max: -${stats.max_depth_cm || 0}cm)`);
    } catch (err) {
        console.error("Failed compiling road audit:", err);
        showToast(`Road compilation failed: ${err.message}`);
    } finally {
        potholeCompileSpinner.style.display = "none";
        potholeRunCompileBtn.disabled = false;
        potholeRunCompileBtn.style.opacity = "1";
        if (potholeRecompileBtn) {
            potholeRecompileBtn.disabled = false;
            potholeRecompileBtn.textContent = "↻ Re-run";
        }
    }
}

async function toggleRoadPotholesFilter(forceState) {
    const nextState = (forceState !== undefined) ? forceState : !isRoadPotholesActive;
    isRoadPotholesActive = nextState;

    if (potholeFilterBtn) potholeFilterBtn.classList.toggle("active", isRoadPotholesActive);
    if (menuFormatPotholes) menuFormatPotholes.classList.toggle("active", isRoadPotholesActive);
    if (checkFormatPotholes) checkFormatPotholes.textContent = isRoadPotholesActive ? "✓" : "";
    const checkToolPotholes = document.getElementById("check-tool-potholes");
    if (checkToolPotholes) checkToolPotholes.textContent = isRoadPotholesActive ? "✓" : "";
    const menuToolPotholes = document.getElementById("menu-tool-potholes");
    if (menuToolPotholes) menuToolPotholes.classList.toggle("active", isRoadPotholesActive);

    roadPotholeGroup.visible = isRoadPotholesActive;

    if (!isRoadPotholesActive) {
        if (potholeAuditDrawer) potholeAuditDrawer.classList.remove("active");
        if (videoRoadBadge) videoRoadBadge.style.display = "none";
        clearVideoOverlay();
        showToast("Road & Potholes filter deactivated.");
        return;
    }

    // Activated: check road audit status
    try {
        const res = await fetch(`${API_BASE}/api/road/audit?t=${Date.now()}`);
        if (!res.ok) throw new Error("Could not retrieve road audit status");
        roadAuditData = await res.json();

        if (!roadAuditData || roadAuditData.status === "not_compiled" || !roadAuditData.compiled) {
            // Not yet compiled: show on-demand compilation view
            if (potholeCountPill) {
                potholeCountPill.textContent = "0";
                potholeCountPill.style.background = "var(--accent-amber)";
            }
            if (potholeCompileView) potholeCompileView.style.display = "flex";
            if (potholeResultsView) potholeResultsView.style.display = "none";
            if (potholeRecompileBtn) potholeRecompileBtn.style.display = "none";
            if (potholeAuditDrawer) potholeAuditDrawer.classList.add("active");
            showToast("Road & Potholes: On-Demand feature ready to compile.");
            return;
        }

        // Already compiled: render immediately
        render3DPotholeMarkers(roadAuditData.potholes, roadAuditData.raw_model_center);
        renderPotholeDrawer(roadAuditData);

        if (videoRoadBadge && videoRoadBadgeText && videoRoadDot) {
            videoRoadBadge.style.display = "flex";
            const isMuddy = roadAuditData.road_classification && roadAuditData.road_classification.road_type === "MUDDY_UNPAVED";
            videoRoadBadgeText.textContent = isMuddy ? "BRO MUDDY ROAD" : "BRO TAR ROAD";
            videoRoadDot.classList.toggle("muddy", isMuddy);
        }

        updateVideoOverlay();

        const stats = roadAuditData.statistics || {};
        const road = roadAuditData.road_classification || {};
        showToast(`BRO ROAD AUDIT: ${road.road_type_label || 'Corridor'} | ${stats.total_potholes || 0} Potholes (Max: -${stats.max_depth_cm || 0}cm)`);
    } catch (err) {
        console.error("Road audit error:", err);
        showToast(`Failed loading road audit: ${err.message}`);
        isRoadPotholesActive = false;
        if (potholeFilterBtn) potholeFilterBtn.classList.remove("active");
        if (checkFormatPotholes) checkFormatPotholes.textContent = "";
        const ctp = document.getElementById("check-tool-potholes");
        if (ctp) ctp.textContent = "";
        const mtp = document.getElementById("menu-tool-potholes");
        if (mtp) mtp.classList.remove("active");
        roadPotholeGroup.visible = false;
    }
}

// Event Listeners for Road & Potholes
if (potholeFilterBtn) {
    potholeFilterBtn.addEventListener("click", () => {
        toggleRoadPotholesFilter();
    });
}

if (menuFormatPotholes) {
    menuFormatPotholes.addEventListener("click", () => {
        closeAllDropdowns();
        toggleRoadPotholesFilter();
    });
}

if (closePotholeDrawerBtn) {
    closePotholeDrawerBtn.addEventListener("click", () => {
        toggleRoadPotholesFilter(false);
    });
}

if (potholeRunCompileBtn) {
    potholeRunCompileBtn.addEventListener("click", () => {
        executeRoadAuditCompilation();
    });
}

if (potholeRecompileBtn) {
    potholeRecompileBtn.addEventListener("click", () => {
        executeRoadAuditCompilation();
    });
}

if (toggleVideoOverlayBtn) {
    toggleVideoOverlayBtn.addEventListener("click", () => {
        isVideoOverlayActive = !isVideoOverlayActive;
        toggleVideoOverlayBtn.textContent = isVideoOverlayActive ? "Vision: ON" : "Vision: OFF";
        toggleVideoOverlayBtn.classList.toggle("active", isVideoOverlayActive);
        if (!isVideoOverlayActive) clearVideoOverlay();
        else updateVideoOverlay();
    });
}

// =================================================================
// 14. AIRSPACE & TERRAIN HEIGHT RESTRICTION AUDIT ENGINE
// =================================================================
let isHeightRestrictionActive = false;
let currentHeightCeilingM = 5.0;
let currentHeightDatumMode = "ground"; // "ground" or "origin"
let heightRestrictionGroup = new THREE.Group();
scene.add(heightRestrictionGroup);

let heightPlaneMesh = null;
let heightGridHelper = null;
let heightBorderLine = null;
let heightPeakStemLine = null;
let heightPeakBeaconMesh = null;
let heightPeakSprite = null;
let highestPeakPoint = null;
let highestPeakHeightM = 0;
let lastViolatingPointsCount = 0;
let totalScenePointsCount = 0;

// Helper: 3D Billboard Sprite Label for Peak Breach Beacon
function createTacticalTextSprite(text, bgColor = "rgba(220, 38, 38, 0.88)") {
    const canvas = document.createElement("canvas");
    canvas.width = 440;
    canvas.height = 100;
    const ctx = canvas.getContext("2d");

    // Tactical Pill Background
    ctx.fillStyle = bgColor;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 3;
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(10, 10, 420, 80, 10);
        ctx.fill();
        ctx.stroke();
    } else {
        ctx.fillRect(10, 10, 420, 80);
        ctx.strokeRect(10, 10, 420, 80);
    }

    ctx.font = "bold 26px 'Inter', sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 220, 50);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(4.0, 0.95, 1.0);
    return sprite;
}

// Helper: Invariant Metric Scale Factor (GPS/telemetry synced)
function getSceneMetricScale() {
    if (flightData && typeof flightData.metric_scale_factor === "number" && flightData.metric_scale_factor > 0) {
        return flightData.metric_scale_factor;
    }
    const modelBox = new THREE.Box3().setFromObject(tacticalModelGroup);
    const modelSize = modelBox.getSize(new THREE.Vector3());
    const modelLength = Math.max(modelSize.x, modelSize.z) || 1.0;
    const realWorldFlightDistance = (flightData && flightData.total_distance_meters) ? flightData.total_distance_meters : 125.0;
    return realWorldFlightDistance / modelLength;
}

// Helper: Fast scene elevation span estimator
function computeSceneElevationRange() {
    if (!pointsNode || !pointsNode.geometry) {
        return { groundY: 0, peakY: 0, heightSpanM: 0, metricScale: 1.0 };
    }
    const posAttr = pointsNode.geometry.getAttribute("position");
    if (!posAttr) return { groundY: 0, peakY: 0, heightSpanM: 0, metricScale: 1.0 };

    tacticalModelGroup.updateMatrixWorld(true);
    const m = tacticalModelGroup.matrixWorld.elements;
    const count = posAttr.count;
    const posArr = posAttr.array;
    const step = Math.max(1, Math.floor(count / 3000));
    const sampledY = [];

    for (let i = 0; i < count; i += step) {
        const idx = i * 3;
        const x = posArr[idx];
        const y = posArr[idx + 1];
        const z = posArr[idx + 2];
        const wy = m[1] * x + m[5] * y + m[9] * z + m[13];
        sampledY.push(wy);
    }
    sampledY.sort((a, b) => a - b);

    const groundY = sampledY[Math.floor(sampledY.length * 0.02)] || 0;
    const peakY = sampledY[Math.min(sampledY.length - 1, Math.floor(sampledY.length * 0.995))] || 0;
    const metricScale = getSceneMetricScale();
    const heightSpanM = Math.max(0, (peakY - groundY) * metricScale);

    return { groundY, peakY, heightSpanM, metricScale };
}

// Open Height Restriction Configuration Modal
function openHeightRestrictionModal() {
    const modal = document.getElementById("height-restriction-modal");
    if (!modal) return;

    const numInput = document.getElementById("height-number-input");
    if (numInput) numInput.value = currentHeightCeilingM;

    document.querySelectorAll(".height-preset-btn").forEach(btn => {
        const h = parseFloat(btn.dataset.height);
        btn.classList.toggle("active", Math.abs(h - currentHeightCeilingM) < 0.05);
    });

    modal.classList.add("active");
}

// Close Height Restriction Modal
function closeHeightRestrictionModal() {
    const modal = document.getElementById("height-restriction-modal");
    if (modal) modal.classList.remove("active");
}

// Core Engine: Calculate Heights, Re-Color Points & Render Ceiling
function applyHeightRestriction(maxHeightM, datumMode = "ground", isLiveUpdate = false) {
    if (!pointsNode || !pointsNode.geometry) {
        showToast("No active 3D model loaded to apply height restriction.");
        return;
    }

    currentHeightCeilingM = Math.max(0.1, parseFloat(maxHeightM) || 5.0);
    currentHeightDatumMode = datumMode;
    isHeightRestrictionActive = true;

    const geo = pointsNode.geometry;
    const posAttr = geo.getAttribute("position");
    if (!posAttr) return;
    const count = posAttr.count;
    totalScenePointsCount = count;

    // Cache original point cloud colors once
    if (!geo.userData.originalColors) {
        if (geo.hasAttribute("color")) {
            geo.userData.originalColors = new Float32Array(geo.getAttribute("color").array);
        } else {
            const defCols = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) {
                defCols[i * 3] = 0.70;
                defCols[i * 3 + 1] = 0.70;
                defCols[i * 3 + 2] = 0.70;
            }
            geo.userData.originalColors = defCols;
            geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(defCols), 3));
        }
    }

    let colAttr = geo.getAttribute("color");
    if (!colAttr) {
        colAttr = new THREE.BufferAttribute(new Float32Array(count * 3), 3);
        geo.setAttribute("color", colAttr);
    }

    const origCols = geo.userData.originalColors;
    const curCols = colAttr.array;

    tacticalModelGroup.updateMatrixWorld(true);
    const m = tacticalModelGroup.matrixWorld.elements;
    const metricScale = getSceneMetricScale();

    // 1. Determine Ground Elevation Baseline
    let groundY = 0;
    const posArr = posAttr.array;
    if (datumMode === "ground") {
        const sampleStep = Math.max(1, Math.floor(count / 2500));
        const sampledY = [];
        for (let i = 0; i < count; i += sampleStep) {
            const sIdx = i * 3;
            const x = posArr[sIdx];
            const y = posArr[sIdx + 1];
            const z = posArr[sIdx + 2];
            sampledY.push(m[1] * x + m[5] * y + m[9] * z + m[13]);
        }
        sampledY.sort((a, b) => a - b);
        groundY = sampledY[Math.floor(sampledY.length * 0.02)] || 0;
    }

    // 2. World Y elevation corresponding to permissible height ceiling
    const ceilingWorldY = groundY + (currentHeightCeilingM / metricScale);

    let violatingCount = 0;
    let maxWorldY = groundY;
    let peakWorldPos = new THREE.Vector3();

    // 3. Mark points exceeding ceiling in Glowing Alert Red/Crimson
    // KEEP original natural photorealistic colors for compliant points!
    for (let i = 0; i < count; i++) {
        const idx = i * 3;
        const x = posArr[idx];
        const y = posArr[idx + 1];
        const z = posArr[idx + 2];

        const worldY = m[1] * x + m[5] * y + m[9] * z + m[13];

        if (worldY > ceilingWorldY) {
            violatingCount++;
            if (worldY > maxWorldY) {
                maxWorldY = worldY;
                const worldX = m[0] * x + m[4] * y + m[8] * z + m[12];
                const worldZ = m[2] * x + m[6] * y + m[10] * z + m[14];
                peakWorldPos.set(worldX, worldY, worldZ);
            }

            // Exceeds restriction: High-Visibility Radiant Crimson
            curCols[idx] = 1.0;
            curCols[idx + 1] = 0.06;
            curCols[idx + 2] = 0.18;
        } else {
            // Compliant point: PRESERVE ORIGINAL NATURAL COLORS (Do not grey out or wipe colors!)
            curCols[idx] = origCols[idx];
            curCols[idx + 1] = origCols[idx + 1];
            curCols[idx + 2] = origCols[idx + 2];
        }
    }

    colAttr.needsUpdate = true;
    if (pointsMaterial) {
        pointsMaterial.vertexColors = true;
        pointsMaterial.needsUpdate = true;
    }

    // 4. Also Colorize Solid Mesh Vertices if active
    if (solidMesh && solidMesh.geometry && solidMesh.geometry.attributes.position) {
        const meshGeo = solidMesh.geometry;
        const meshPos = meshGeo.getAttribute("position");
        const meshCount = meshPos.count;
        const meshPosArr = meshPos.array;

        if (!meshGeo.userData.originalColors) {
            if (meshGeo.hasAttribute("color")) {
                meshGeo.userData.originalColors = new Float32Array(meshGeo.getAttribute("color").array);
            } else {
                const defMeshCols = new Float32Array(meshCount * 3);
                for (let k = 0; k < meshCount * 3; k++) defMeshCols[k] = 0.75;
                meshGeo.userData.originalColors = defMeshCols;
                meshGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(defMeshCols), 3));
            }
        }
        let meshColAttr = meshGeo.getAttribute("color");
        if (!meshColAttr) {
            meshColAttr = new THREE.BufferAttribute(new Float32Array(meshCount * 3), 3);
            meshGeo.setAttribute("color", meshColAttr);
        }
        const meshCols = meshColAttr.array;
        const meshOrig = meshGeo.userData.originalColors;

        for (let i = 0; i < meshCount; i++) {
            const midx = i * 3;
            const mx = meshPosArr[midx];
            const my = meshPosArr[midx + 1];
            const mz = meshPosArr[midx + 2];
            const mwy = m[1] * mx + m[5] * my + m[9] * mz + m[13];

            if (mwy > ceilingWorldY) {
                meshCols[midx] = 1.0;
                meshCols[midx + 1] = 0.06;
                meshCols[midx + 2] = 0.18;
            } else {
                // Preserve original mesh color
                meshCols[midx] = meshOrig[midx];
                meshCols[midx + 1] = meshOrig[midx + 1];
                meshCols[midx + 2] = meshOrig[midx + 2];
            }
        }
        meshColAttr.needsUpdate = true;
        if (standardMaterial) {
            standardMaterial.vertexColors = true;
            standardMaterial.needsUpdate = true;
        }
    }

    // 5. Compute Stats & Metrics
    const peakElevM = (maxWorldY - groundY) * metricScale;
    const maxBreachM = Math.max(0, peakElevM - currentHeightCeilingM);
    highestPeakPoint = violatingCount > 0 ? peakWorldPos.clone() : null;
    highestPeakHeightM = peakElevM;
    lastViolatingPointsCount = violatingCount;

    // 6. Build or Update 3D Ceiling Plane & Beacon
    renderHeightCeiling3D(ceilingWorldY, peakWorldPos, maxBreachM, peakElevM, violatingCount);

    // 7. Update Drawer & UI Stats
    updateHeightDrawerUI(count, violatingCount, peakElevM, maxBreachM, groundY * metricScale);

    // 8. Toolbar & Menu UI Sync
    const clearMenuBtn = document.getElementById("menu-tool-clear-height");
    if (clearMenuBtn) clearMenuBtn.style.display = "flex";

    const toolbarBtn = document.getElementById("height-restriction-btn");
    if (toolbarBtn) {
        toolbarBtn.style.background = "rgba(245, 158, 11, 0.2)";
        toolbarBtn.style.borderColor = "var(--accent-amber)";
        toolbarBtn.innerHTML = `▲ Ceiling: ${currentHeightCeilingM.toFixed(1)}m`;
    }

    if (!isLiveUpdate) {
        closeHeightRestrictionModal();
        showToast(`HEIGHT AUDIT: Ceiling &le; ${currentHeightCeilingM.toFixed(1)}m | ${violatingCount.toLocaleString()} points marked exceeding limit.`);
    }
}

// 3D Visual Ceiling Plane, Grid, Border Lines & Peak Beacon
function renderHeightCeiling3D(ceilingWorldY, peakWorldPos, maxBreachM, peakElevM, violatingCount) {
    while (heightRestrictionGroup.children.length > 0) {
        const obj = heightRestrictionGroup.children[0];
        heightRestrictionGroup.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
            else obj.material.dispose();
        }
    }

    const box = new THREE.Box3().setFromObject(tacticalModelGroup);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const spanX = Math.max(size.x * 1.35, 12);
    const spanZ = Math.max(size.z * 1.35, 12);

    // 1. Semi-Transparent Restriction Ceiling Plane
    const planeGeo = new THREE.PlaneGeometry(spanX, spanZ);
    planeGeo.rotateX(-Math.PI / 2);
    const planeMat = new THREE.MeshBasicMaterial({
        color: 0xef4444,
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    heightPlaneMesh = new THREE.Mesh(planeGeo, planeMat);
    heightPlaneMesh.position.set(center.x, ceilingWorldY, center.z);
    heightRestrictionGroup.add(heightPlaneMesh);

    // 2. Tactical Red/Amber Ceiling Grid
    const gridDim = Math.max(spanX, spanZ);
    heightGridHelper = new THREE.GridHelper(gridDim, 24, 0xf59e0b, 0xef4444);
    heightGridHelper.position.set(center.x, ceilingWorldY, center.z);
    heightGridHelper.material.opacity = 0.35;
    heightGridHelper.material.transparent = true;
    heightRestrictionGroup.add(heightGridHelper);

    // 3. Perimeter Warning Border Line
    const halfX = spanX / 2;
    const halfZ = spanZ / 2;
    const borderPoints = [
        new THREE.Vector3(center.x - halfX, ceilingWorldY, center.z - halfZ),
        new THREE.Vector3(center.x + halfX, ceilingWorldY, center.z - halfZ),
        new THREE.Vector3(center.x + halfX, ceilingWorldY, center.z + halfZ),
        new THREE.Vector3(center.x - halfX, ceilingWorldY, center.z + halfZ),
        new THREE.Vector3(center.x - halfX, ceilingWorldY, center.z - halfZ)
    ];
    const borderGeo = new THREE.BufferGeometry().setFromPoints(borderPoints);
    const borderMat = new THREE.LineBasicMaterial({ color: 0xf59e0b, linewidth: 2 });
    heightBorderLine = new THREE.Line(borderGeo, borderMat);
    heightRestrictionGroup.add(heightBorderLine);

    // 4. Peak Breach Marker, Stem & Billboard Label
    if (violatingCount > 0 && maxBreachM > 0.05) {
        const stemGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(peakWorldPos.x, ceilingWorldY, peakWorldPos.z),
            new THREE.Vector3(peakWorldPos.x, peakWorldPos.y, peakWorldPos.z)
        ]);
        const stemMat = new THREE.LineDashedMaterial({
            color: 0xff0044,
            dashSize: 0.15,
            gapSize: 0.08,
            linewidth: 2
        });
        heightPeakStemLine = new THREE.Line(stemGeo, stemMat);
        heightPeakStemLine.computeLineDistances();
        heightRestrictionGroup.add(heightPeakStemLine);

        // Glowing Beacon Sphere
        const beaconGeo = new THREE.SphereGeometry(0.18, 16, 16);
        const beaconMat = new THREE.MeshBasicMaterial({ color: 0xff0033 });
        heightPeakBeaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
        heightPeakBeaconMesh.position.copy(peakWorldPos);
        heightRestrictionGroup.add(heightPeakBeaconMesh);

        // 3D Billboard Sprite Tag
        const tagText = `▲ PEAK: +${maxBreachM.toFixed(1)}m OVER CEILING`;
        heightPeakSprite = createTacticalTextSprite(tagText, "rgba(220, 38, 38, 0.90)");
        heightPeakSprite.position.set(peakWorldPos.x, peakWorldPos.y + 0.45, peakWorldPos.z);
        heightRestrictionGroup.add(heightPeakSprite);
    }
}

// Update Height Restriction Results Drawer UI
function updateHeightDrawerUI(totalCount, violatingCount, peakElevM, maxBreachM, groundElevM) {
    const drawer = document.getElementById("height-restriction-drawer");
    if (!drawer) return;
    drawer.classList.add("active");

    // Close overlapping drawers on the right
    const potholeDrawer = document.getElementById("pothole-audit-drawer");
    if (potholeDrawer) potholeDrawer.classList.remove("active");

    const pill = document.getElementById("height-violation-pill");
    const ratio = totalCount > 0 ? ((violatingCount / totalCount) * 100).toFixed(1) : "0.0";
    if (pill) {
        pill.textContent = `${violatingCount.toLocaleString()} PTS (${ratio}%)`;
        pill.style.background = violatingCount > 0 ? "var(--accent-red)" : "var(--accent-green)";
        pill.style.color = "#fff";
    }

    const ceilingLabel = document.getElementById("height-ceiling-label");
    if (ceilingLabel) ceilingLabel.textContent = `Ceiling: ≤ ${currentHeightCeilingM.toFixed(1)}m`;

    const peakLabel = document.getElementById("height-peak-label");
    if (peakLabel) peakLabel.textContent = `Peak: ${peakElevM.toFixed(1)}m`;

    const breachLabel = document.getElementById("height-breach-label");
    if (breachLabel) {
        breachLabel.textContent = violatingCount > 0 ? `Breach: +${maxBreachM.toFixed(1)}m` : `Breach: 0.0m`;
        breachLabel.style.color = violatingCount > 0 ? "var(--accent-red)" : "var(--accent-green)";
    }

    const slider = document.getElementById("height-live-slider");
    if (slider) slider.value = currentHeightCeilingM;

    const liveInput = document.getElementById("height-live-input");
    if (liveInput && document.activeElement !== liveInput) {
        liveInput.value = currentHeightCeilingM.toFixed(1);
    }

    const statTotal = document.getElementById("height-stat-total");
    if (statTotal) statTotal.textContent = totalCount.toLocaleString();

    const statViolating = document.getElementById("height-stat-violating");
    if (statViolating) statViolating.textContent = violatingCount.toLocaleString();

    const statRatio = document.getElementById("height-stat-ratio");
    if (statRatio) statRatio.textContent = `${ratio}%`;

    const statGround = document.getElementById("height-stat-ground");
    if (statGround) statGround.textContent = `${groundElevM.toFixed(1)}m`;

    const statusCard = document.getElementById("height-status-card");
    const statusTitle = document.getElementById("height-status-title");
    const statusSub = document.getElementById("height-status-sub");
    if (statusCard && statusTitle && statusSub) {
        if (violatingCount > 0) {
            statusCard.style.background = "rgba(239, 68, 68, 0.12)";
            statusCard.style.borderColor = "rgba(239, 68, 68, 0.3)";
            statusCard.style.color = "#f85149";
            statusTitle.textContent = "RESTRICTION BREACH DETECTED";
            statusSub.textContent = `${violatingCount.toLocaleString()} points exceed the vertical ceiling limit. Marked in high-visibility Crimson.`;
        } else {
            statusCard.style.background = "rgba(16, 185, 129, 0.12)";
            statusCard.style.borderColor = "rgba(16, 185, 129, 0.3)";
            statusCard.style.color = "var(--accent-green)";
            statusTitle.textContent = "AIRSPACE & ELEVATION COMPLIANT";
            statusSub.textContent = `All scene points and structures are completely within the ${currentHeightCeilingM.toFixed(1)}m restriction ceiling.`;
        }
    }
}

// Clear Height Restriction & Restore Original Colors
function clearHeightRestriction() {
    isHeightRestrictionActive = false;

    // Restore point cloud original colors
    if (pointsNode && pointsNode.geometry && pointsNode.geometry.userData.originalColors) {
        const colAttr = pointsNode.geometry.getAttribute("color");
        if (colAttr) {
            colAttr.array.set(pointsNode.geometry.userData.originalColors);
            colAttr.needsUpdate = true;
        }
    }

    // Restore mesh original colors
    if (solidMesh && solidMesh.geometry && solidMesh.geometry.userData.originalColors) {
        const mCol = solidMesh.geometry.getAttribute("color");
        if (mCol) {
            mCol.array.set(solidMesh.geometry.userData.originalColors);
            mCol.needsUpdate = true;
        }
    }

    // Remove 3D visual ceiling objects
    while (heightRestrictionGroup.children.length > 0) {
        const obj = heightRestrictionGroup.children[0];
        heightRestrictionGroup.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
            else obj.material.dispose();
        }
    }

    // Hide drawer
    const drawer = document.getElementById("height-restriction-drawer");
    if (drawer) drawer.classList.remove("active");

    // Reset toolbar button
    const toolbarBtn = document.getElementById("height-restriction-btn");
    if (toolbarBtn) {
        toolbarBtn.style.background = "";
        toolbarBtn.style.borderColor = "rgba(245, 158, 11, 0.4)";
        toolbarBtn.innerHTML = `▲ Height Restriction`;
    }

    // Hide clear menu item
    const clearMenuBtn = document.getElementById("menu-tool-clear-height");
    if (clearMenuBtn) clearMenuBtn.style.display = "none";

    showToast("Height restriction cleared. Original scene restored.");
}

// Focus Camera on Peak Breach
function focusOnPeakObstacle() {
    if (!highestPeakPoint) {
        showToast("No structure currently exceeds the height ceiling.");
        return;
    }
    const target = highestPeakPoint.clone();
    controls.target.copy(target);
    camera.position.set(target.x + 4, target.y + 3, target.z + 5);
    controls.update();
    showToast(`Camera focused on peak obstacle (${highestPeakHeightM.toFixed(1)}m elevation).`);
}

// Export Audit Report
function exportHeightRestrictionReport() {
    const report = {
        title: "PRISM-3D // AIRSPACE & TERRAIN HEIGHT RESTRICTION AUDIT REPORT",
        timestamp: new Date().toISOString(),
        model: "actionable_threat_map.ply",
        restriction_ceiling_meters: currentHeightCeilingM,
        datum_reference: currentHeightDatumMode,
        peak_detected_elevation_meters: Number(highestPeakHeightM.toFixed(2)),
        max_breach_meters: Number(Math.max(0, highestPeakHeightM - currentHeightCeilingM).toFixed(2)),
        violating_points_count: lastViolatingPointsCount,
        total_model_points: totalScenePointsCount,
        violation_ratio_percent: totalScenePointsCount > 0 ? Number(((lastViolatingPointsCount / totalScenePointsCount) * 100).toFixed(2)) : 0,
        compliance_status: lastViolatingPointsCount > 0 ? "BREACH_DETECTED" : "COMPLIANT",
        peak_coordinates: highestPeakPoint ? {
            x: Number(highestPeakPoint.x.toFixed(3)),
            y: Number(highestPeakPoint.y.toFixed(3)),
            z: Number(highestPeakPoint.z.toFixed(3))
        } : null
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `PRISM_Height_Restriction_Report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Height restriction audit report exported successfully.");
}

// =================================================================
// 15. HEIGHT RESTRICTION EVENT LISTENERS & WIRING
// =================================================================
// Viewport Toolbar Button
const heightRestrictionBtn = document.getElementById("height-restriction-btn");
if (heightRestrictionBtn) {
    heightRestrictionBtn.addEventListener("click", () => {
        openHeightRestrictionModal();
    });
}

// Modal Controls
const closeHeightModalBtn = document.getElementById("close-height-modal-btn");
if (closeHeightModalBtn) {
    closeHeightModalBtn.addEventListener("click", closeHeightRestrictionModal);
}

const cancelHeightModalBtn = document.getElementById("cancel-height-modal-btn");
if (cancelHeightModalBtn) {
    cancelHeightModalBtn.addEventListener("click", closeHeightRestrictionModal);
}

const confirmHeightRestrictionBtn = document.getElementById("confirm-height-restriction-btn");
if (confirmHeightRestrictionBtn) {
    confirmHeightRestrictionBtn.addEventListener("click", () => {
        const numInput = document.getElementById("height-number-input");
        const val = numInput ? parseFloat(numInput.value) : 5.0;
        applyHeightRestriction(val, currentHeightDatumMode, false);
    });
}

// Preset Buttons
document.querySelectorAll(".height-preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".height-preset-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const val = parseFloat(btn.dataset.height);
        const numInput = document.getElementById("height-number-input");
        if (numInput) numInput.value = val;
    });
});

// Number Input Change
const heightNumberInput = document.getElementById("height-number-input");
if (heightNumberInput) {
    heightNumberInput.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value) || 0;
        document.querySelectorAll(".height-preset-btn").forEach(b => {
            b.classList.toggle("active", Math.abs(parseFloat(b.dataset.height) - val) < 0.05);
        });
    });
    heightNumberInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            const val = parseFloat(heightNumberInput.value) || 5.0;
            applyHeightRestriction(val, "ground", false);
        }
    });
}

// Drawer Controls
const closeHeightDrawerBtn = document.getElementById("close-height-drawer-btn");
if (closeHeightDrawerBtn) {
    closeHeightDrawerBtn.addEventListener("click", () => {
        const drawer = document.getElementById("height-restriction-drawer");
        if (drawer) drawer.classList.remove("active");
    });
}

const heightReconfigBtn = document.getElementById("height-reconfig-btn");
if (heightReconfigBtn) {
    heightReconfigBtn.addEventListener("click", openHeightRestrictionModal);
}

const heightLiveSlider = document.getElementById("height-live-slider");
const heightLiveInput = document.getElementById("height-live-input");

if (heightLiveSlider) {
    heightLiveSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        if (heightLiveInput && document.activeElement !== heightLiveInput) {
            heightLiveInput.value = val.toFixed(1);
        }
        applyHeightRestriction(val, "ground", true);
    });
}

if (heightLiveInput) {
    heightLiveInput.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        if (!isNaN(val) && val > 0) {
            if (heightLiveSlider) {
                heightLiveSlider.value = Math.min(30.0, Math.max(0.5, val));
            }
            applyHeightRestriction(val, "ground", true);
        }
    });
    heightLiveInput.addEventListener("change", (e) => {
        const val = parseFloat(e.target.value);
        if (!isNaN(val) && val > 0) {
            if (heightLiveSlider) {
                heightLiveSlider.value = Math.min(30.0, Math.max(0.5, val));
            }
            applyHeightRestriction(val, "ground", true);
        }
    });
}

const heightFocusPeakBtn = document.getElementById("height-focus-peak-btn");
if (heightFocusPeakBtn) {
    heightFocusPeakBtn.addEventListener("click", focusOnPeakObstacle);
}

const heightTogglePlaneBtn = document.getElementById("height-toggle-plane-btn");
if (heightTogglePlaneBtn) {
    heightTogglePlaneBtn.addEventListener("click", () => {
        heightRestrictionGroup.visible = !heightRestrictionGroup.visible;
        heightTogglePlaneBtn.textContent = heightRestrictionGroup.visible ? "Grid: ON" : "Grid: OFF";
        heightTogglePlaneBtn.classList.toggle("active", heightRestrictionGroup.visible);
    });
}

const heightExportReportBtn = document.getElementById("height-export-report-btn");
if (heightExportReportBtn) {
    heightExportReportBtn.addEventListener("click", exportHeightRestrictionReport);
}

const heightClearBtn = document.getElementById("height-clear-btn");
if (heightClearBtn) {
    heightClearBtn.addEventListener("click", clearHeightRestriction);
}