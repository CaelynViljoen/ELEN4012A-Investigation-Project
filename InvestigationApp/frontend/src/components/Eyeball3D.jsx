// src/components/Eyeball3D.jsx

import React, { Suspense, useRef, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';

const EyeballModel = () => {
  const ref = useRef();
  const lightRef = useRef();
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // Load the 3D model of the eyeball
  const { scene } = useGLTF('/assets/eyeball2.glb');

  // Update rotation based on mouse position
  useEffect(() => {
    const handleMouseMove = (event) => {
      if (ref.current) {
        const { innerWidth, innerHeight } = window;

        // Define the offset based on the quadrant positioning of the eyeball (e.g., bottom left quadrant)
        const offsetX = innerWidth / 4; // Adjust this value based on the eyeball's horizontal offset
        const offsetY = -innerHeight / 4; // Adjust this value based on the eyeball's vertical offset

        // Calculate the normalized mouse position relative to the eyeball's offset
        const x = ((event.clientX + offsetX) / innerWidth) * 3 - 1;
        const y = -((event.clientY + offsetY) / innerHeight) * 2 + 1;

        // Update rotation based on adjusted mouse position
        ref.current.rotation.y = x * 0.5; // Horizontal movement affects y-rotation
        ref.current.rotation.x = -y * 0.5; // Correct vertical movement inversion

        // Update the state with the mouse position
        setMousePosition({ x: event.clientX, y: event.clientY });
      }
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  // Position the spotlight in front and to the right of the eyeball
  useFrame(() => {
    if (lightRef.current) {
      lightRef.current.position.set(2, 1, 3); // Position the light slightly to the right and in front of the eyeball
      lightRef.current.target.position.set(0, 0, 0); // Point the light directly at the eyeball
      lightRef.current.target.updateMatrixWorld();
    }
  });

  return (
    <>
      <primitive object={scene} ref={ref} scale={[1.5, 1.5, 1.5]} /> {/* Scale up the eyeball */}
      <spotLight
        ref={lightRef}
        intensity={35} // Increase intensity for better illumination
        angle={0.5}
        penumbra={0.5}
        castShadow
        distance={10}
      />
    </>
  );
};

const Eyeball3D = () => {
  return (
    <Canvas
      camera={{ position: [0, 0, 5], fov: 50, near: 0.1, far: 1000 }}
      style={{ background: 'transparent' }} // Set canvas background to transparent
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} />
      <Suspense fallback={null}>
        <EyeballModel />
      </Suspense>
      <OrbitControls enablePan={false} enableZoom={false} />
    </Canvas>
  );
};

export default Eyeball3D;
