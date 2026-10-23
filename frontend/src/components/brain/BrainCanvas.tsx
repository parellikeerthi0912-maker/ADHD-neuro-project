import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { BrainRegionKey } from '../../types';
import { BrainFallback } from './BrainFallback';
import brainIsolatedUrl from '../../assets/brain_isolated.png';

interface BrainCanvasProps {
  activeRegion?: BrainRegionKey | null;
  onRegionHover?: (region: BrainRegionKey | null) => void;
  onRegionClick?: (region: BrainRegionKey) => void;
  isOpeningCycleActive?: boolean;
  interactive?: boolean;
  introDuration?: number;
}

export const BrainCanvas: React.FC<BrainCanvasProps> = ({
  activeRegion,
  onRegionHover,
  onRegionClick,
  isOpeningCycleActive = true,
  interactive = true,
  introDuration = 4.0,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mouseRef = useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });
  const activeRegionRef = useRef<BrainRegionKey | null>(activeRegion ?? null);
  const [webGlAvailable, setWebGlAvailable] = useState<boolean>(true);

  useEffect(() => {
    activeRegionRef.current = activeRegion ?? null;
  }, [activeRegion]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    try {
      const testCanvas = document.createElement('canvas');
      const gl = testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl');
      if (!gl) {
        setWebGlAvailable(false);
        return;
      }
    } catch {
      setWebGlAvailable(false);
      return;
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // =========================================================================
    // 1. SCENE, CAMERA, RENDERER
    // =========================================================================
    const scene = new THREE.Scene();

    let width = container.clientWidth || window.innerWidth;
    let height = container.clientHeight || window.innerHeight;

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 0, 7.0);

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance',
      });
    } catch {
      setWebGlAvailable(false);
      return;
    }

    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4;
    container.appendChild(renderer.domElement);

    // =========================================================================
    // 2. CINEMATIC MULTI-POINT LIGHTING
    // =========================================================================
    const ambientLight = new THREE.AmbientLight(0x0e142a, 2.5);
    scene.add(ambientLight);

    // Front Left Cyan Light
    const cyanLight = new THREE.PointLight(0x00e5ff, 5.0, 15);
    cyanLight.position.set(-3.2, 2.0, 4.0);
    scene.add(cyanLight);

    // Front Right Magenta Light
    const magentaLight = new THREE.PointLight(0xf02bd5, 5.5, 15);
    magentaLight.position.set(3.2, -1.2, 4.0);
    scene.add(magentaLight);

    // Top Violet Light
    const violetLight = new THREE.PointLight(0x8b5cf6, 4.5, 18);
    violetLight.position.set(0, 3.8, -2.0);
    scene.add(violetLight);

    // Active Region Flare Spotlight
    const regionLight = new THREE.PointLight(0x00e5ff, 0, 10);
    scene.add(regionLight);

    // =========================================================================
    // 3. EXACT REFERENCE BRAIN MESH (Curved 3D Volumetric Stage)
    // =========================================================================
    const brainMasterGroup = new THREE.Group();
    scene.add(brainMasterGroup);

    const textureLoader = new THREE.TextureLoader();
    const brainTexture = textureLoader.load(brainIsolatedUrl);
    brainTexture.generateMipmaps = true;
    brainTexture.minFilter = THREE.LinearMipmapLinearFilter;
    brainTexture.magFilter = THREE.LinearFilter;

    // Physical dimensions matching square aspect ratio 1024x1024
    const planeW = 4.2;
    const planeH = 4.2;

    // Crisp flat plane geometry matching aspect ratio
    const frontGeo = new THREE.PlaneGeometry(planeW, planeH);

    const frontMat = new THREE.MeshBasicMaterial({
      map: brainTexture,
      transparent: true,
      side: THREE.FrontSide,
      depthWrite: false,
    });
    const frontMesh = new THREE.Mesh(frontGeo, frontMat);
    brainMasterGroup.add(frontMesh);

    // =========================================================================
    // 4. ORBITING 3D ELECTRICAL PARTICLES
    // =========================================================================
    const pulseCount = 36;
    const pulseMeshes: THREE.Mesh[] = [];
    const pulseSpeedOffsets: number[] = [];
    const pulseRadii: number[] = [];
    const pulseYOffsets: number[] = [];
    const pulseColors = [0x00e5ff, 0x287bff, 0x8b5cf6, 0xf02bd5, 0xff65d4, 0x35f5c8];

    const pulseGeo = new THREE.SphereGeometry(0.045, 8, 8);

    for (let p = 0; p < pulseCount; p++) {
      const col = pulseColors[p % pulseColors.length];
      const pMat = new THREE.MeshBasicMaterial({
        color: col,
        blending: THREE.AdditiveBlending,
      });
      const mesh = new THREE.Mesh(pulseGeo, pMat);
      brainMasterGroup.add(mesh);
      pulseMeshes.push(mesh);

      pulseSpeedOffsets.push((p / pulseCount) * Math.PI * 2);
      pulseRadii.push(2.1 + (p % 4) * 0.35);
      pulseYOffsets.push(-0.6 + (p % 6) * 0.35);
    }

    // =========================================================================
    // 5. RISING HOLOGRAPHIC SPARKS
    // =========================================================================
    const pedestalGroup = new THREE.Group();
    pedestalGroup.position.y = -1.65;
    scene.add(pedestalGroup);

    // Rising Sparks from Holographic Pedestal
    const sparkCount = 40;
    const sparkPos = new Float32Array(sparkCount * 3);
    const sparkYOffsets: number[] = [];
    for (let s = 0; s < sparkCount; s++) {
      const rad = 0.5 + Math.random() * 1.6;
      const theta = Math.random() * Math.PI * 2;
      sparkPos[s * 3] = Math.cos(theta) * rad;
      sparkPos[s * 3 + 1] = Math.random() * 2.2;
      sparkPos[s * 3 + 2] = Math.sin(theta) * rad;
      sparkYOffsets.push(0.01 + Math.random() * 0.015);
    }
    const sparkGeo = new THREE.BufferGeometry();
    sparkGeo.setAttribute('position', new THREE.BufferAttribute(sparkPos, 3));
    const sparkMat = new THREE.PointsMaterial({
      color: 0x00e5ff,
      size: 0.07,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
    });
    const sparkPoints = new THREE.Points(sparkGeo, sparkMat);
    pedestalGroup.add(sparkPoints);

    // =========================================================================
    // 6. MOUSE & RESIZE LISTENERS
    // =========================================================================
    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
      mouseRef.current.targetX = x * 0.65;
      mouseRef.current.targetY = y * 0.5;
    };

    const handleResize = () => {
      if (!container) return;
      width = container.clientWidth || window.innerWidth;
      height = container.clientHeight || window.innerHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    if (interactive) {
      window.addEventListener('mousemove', handleMouseMove);
    }
    window.addEventListener('resize', handleResize);

    // =========================================================================
    // 7. ANIMATION LOOP (Smooth Floating, 3D Tilt, Orbiting Pulses)
    // =========================================================================
    let clock = new THREE.Clock();
    let animId: number;
    let isVisible = true;

    const handleVisibility = () => {
      isVisible = !document.hidden;
    };
    document.addEventListener('visibilitychange', handleVisibility);

    const render = () => {
      animId = requestAnimationFrame(render);
      if (!isVisible) return;

      const delta = clock.getDelta();
      const time = clock.getElapsedTime();

      // Inertia mouse damping
      mouseRef.current.x += (mouseRef.current.targetX - mouseRef.current.x) * 0.07;
      mouseRef.current.y += (mouseRef.current.targetY - mouseRef.current.y) * 0.07;

      if (!prefersReducedMotion) {
        // Continuous Majestic 3D Yaw Rotation & Tilt
        brainMasterGroup.rotation.y = Math.sin(time * 0.65) * 0.28 + mouseRef.current.x * 0.55;
        brainMasterGroup.rotation.x = Math.cos(time * 0.5) * 0.1 - mouseRef.current.y * 0.38;

        // Smooth Sinusoidal Floating Levitation on Y-Axis
        brainMasterGroup.position.y = Math.sin(time * 1.5) * 0.16;

        // Subtle Organic Breathing Scale
        const breathe = 1.0 + Math.sin(time * 2.2) * 0.025;
        brainMasterGroup.scale.set(breathe, breathe, breathe);



        // 3D Orbiting Electrical Pulses
        for (let p = 0; p < pulseCount; p++) {
          const angle = time * 1.35 + pulseSpeedOffsets[p];
          const rad = pulseRadii[p];
          const yBase = pulseYOffsets[p];

          pulseMeshes[p].position.x = Math.cos(angle) * rad;
          pulseMeshes[p].position.z = Math.sin(angle) * (rad * 0.7);
          pulseMeshes[p].position.y = yBase + Math.sin(angle * 2.2) * 0.25;

          const pScale = 0.85 + Math.sin(time * 9 + p) * 0.35;
          pulseMeshes[p].scale.setScalar(pScale);
        }

        // Rising Sparks Float
        const spPos = sparkGeo.attributes.position;
        for (let s = 0; s < sparkCount; s++) {
          let y = spPos.getY(s) + sparkYOffsets[s];
          if (y > 2.3) y = 0.05;
          spPos.setY(s, y);
        }
        spPos.needsUpdate = true;

        // Dynamic Lighting Oscillations
        cyanLight.intensity = 4.8 + Math.sin(time * 2.4) * 1.4;
        magentaLight.intensity = 5.2 + Math.cos(time * 2.0) * 1.4;
      }

      // Camera Glide & Lobe Highlighting
      const targetRegion = activeRegionRef.current;
      const targetCamPos = new THREE.Vector3(0, 0, 7.0);

      if (targetRegion === 'frontal') {
        targetCamPos.set(-0.7, 0.5, 5.8);
        regionLight.intensity = 6.5;
        regionLight.color.setHex(0x00e5ff);
        regionLight.position.set(-1.0, 1.0, 2.0);
      } else if (targetRegion === 'parietal') {
        targetCamPos.set(0.7, 0.7, 5.8);
        regionLight.intensity = 6.5;
        regionLight.color.setHex(0x8b5cf6);
        regionLight.position.set(1.0, 1.0, 2.0);
      } else if (targetRegion === 'temporal') {
        targetCamPos.set(-0.9, -0.4, 5.8);
        regionLight.intensity = 6.5;
        regionLight.color.setHex(0xf02bd5);
        regionLight.position.set(-1.2, -0.5, 2.0);
      } else if (targetRegion === 'occipital') {
        targetCamPos.set(0.9, -0.4, 5.8);
        regionLight.intensity = 6.5;
        regionLight.color.setHex(0x35f5c8);
        regionLight.position.set(1.2, -0.5, 2.0);
      } else {
        regionLight.intensity *= 0.9;
      }

      camera.position.lerp(targetCamPos, 0.045);
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
      document.removeEventListener('visibilitychange', handleVisibility);
      if (interactive) {
        window.removeEventListener('mousemove', handleMouseMove);
      }
      window.removeEventListener('resize', handleResize);
      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [interactive, introDuration]);

  if (!webGlAvailable) {
    return <BrainFallback />;
  }

  return (
    <div
      ref={containerRef}
      className="w-full h-full relative cursor-grab active:cursor-grabbing flex items-center justify-center select-none"
    />
  );
};
