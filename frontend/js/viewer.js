// =================================================================
// PRISM // TACTICAL RECON COMMAND
// VIEWER.JS - 3D POINT-CLOUD INSPECTION & DIGITAL TWIN
// =================================================================

// =================================================================
// 1. LIVE TELEMETRY SYNCHRONIZATION
// =================================================================
let flightData = null;
const video = document.getElementById("uav-video");
const latEl = document.getElementById("lat-val");
const lonEl = document.getElementById("lon-val");
const altEl = document.getElementById("alt-val");
const distEl = document.getElementById("dist-val");

async function fetchTelemetry() {
    try {
        const res = await fetch("http://127.0.0.1:8000/api/telemetry");
        if (!res.ok) throw new Error("Telemetry endpoint unavailable");
        flightData = await res.json();
        console.log("Telemetry loaded:", flightData);
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
    });
}
fetchTelemetry();

// =================================================================
// 2. THREE.JS VIEWPORT & SCENE
// =================================================================
const container = document.getElementById("canvas-container");
if (!container) {
    console.error("ERROR: #canvas-container was not found.");
    throw new Error("#canvas-container does not exist.");
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050608);

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

const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
scene.add(ambientLight);

// =================================================================
// 3. GLOBAL TACTICAL STATE & ORBIT CONTROLS
// =================================================================
let pointCloud = null;
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

// Expose tactical twin context globally for overlay/YOLO integration
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
    if (!pointCloud) return null;
    updateMousePosition(event);
    raycaster.setFromCamera(mouse, camera);
    const intersections = raycaster.intersectObject(pointCloud, false);
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

window.addEventListener("keydown", (e) => {
    const activeElement = document.activeElement;
    if (activeElement && (activeElement.tagName === "INPUT" || activeElement.tagName === "TEXTAREA")) return;

    if (e.key === "m" || e.key === "M") {
        measureMode = !measureMode;
        console.log(`[TACTICAL HUD] Measurement Mode: ${measureMode ? "ARMED" : "STANDBY"}`);
        if (!measureMode) clearMeasurementMarkers();
    }
    
    if (e.key === "f" || e.key === "F") {
        if (selectedPoint && !measureMode) focusOnPoint(selectedPoint, true);
    }
    if (e.key === "r" || e.key === "R") {
        resetCamera();
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

        // 2. Compute scale ratio against flight telemetry
        const modelBox = new THREE.Box3().setFromObject(pointCloud);
        const modelSize = modelBox.getSize(new THREE.Vector3());
        
        const modelLength = Math.max(modelSize.x, modelSize.z); 
        const realWorldFlightDistance = (flightData && flightData.total_distance_meters) ? flightData.total_distance_meters : 125.0;
        const metricScale = realWorldFlightDistance / modelLength;

        // 3. Decompose 3D distance into orthogonal components
        const distanceM = rawDistance * metricScale;
        const heightDiffM = rawHeight * metricScale;
        const horizontalSpanM = Math.sqrt(Math.max(0, (distanceM * distanceM) - (heightDiffM * heightDiffM)));

        console.log(`----------------------------------------`);
        console.log(`SCALE MULTIPLIER: ${metricScale.toFixed(4)}`);
        console.log(`LINE DISTANCE   : ${distanceM.toFixed(2)} m`);
        console.log(`VERT ELEVATION  : ${heightDiffM.toFixed(2)} m`);
        console.log(`HORIZ SPAN      : ${horizontalSpanM.toFixed(2)} m`);
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
// 8. MODEL LOADER & CALIBRATION CONTROLS
// =================================================================
const loader = new THREE.PLYLoader();
const plyPath = "http://127.0.0.1:8000/data/models/actionable_threat_map.ply";

loader.load(
    plyPath,
    (geometry) => {
        console.log("PLY loaded successfully.");
        geometry.computeVertexNormals();

        const material = new THREE.PointsMaterial({
            size: 0.08,
            vertexColors: geometry.hasAttribute("color")
        });

        pointCloud = new THREE.Points(geometry, material);
        geometry.center();

        const positionAttribute = geometry.getAttribute("position");
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

        pointCloud.quaternion.copy(autoAlignPointCloud(samplePoints));
        pointCloud.position.set(0, 0, 0);
        pointCloud.updateMatrixWorld(true);

        const modelBox = new THREE.Box3().setFromObject(pointCloud);
        const modelCenter = modelBox.getCenter(new THREE.Vector3());

        // Center vertically
        pointCloud.position.y = -modelCenter.y;
        pointCloud.updateMatrixWorld(true);

        const finalBox = new THREE.Box3().setFromObject(pointCloud);
        const finalCenter = finalBox.getCenter(new THREE.Vector3());
        const finalSize = finalBox.getSize(new THREE.Vector3());

        initialModelPosition.copy(pointCloud.position);
        initialModelQuaternion.copy(pointCloud.quaternion);

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

        scene.add(pointCloud);
        console.log("Tactical Point Cloud mounted in scene.");

        // Manual Calibration Controls
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
    },
    (xhr) => {
        if (xhr.total) {
            const percent = (xhr.loaded / xhr.total) * 100;
            console.log(`PLY loading: ${percent.toFixed(1)}%`);
        }
    },
    (error) => {
        console.error("Failed to load PLY:", error);
    }
);

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
    renderer.render(scene, camera);
}
animate();