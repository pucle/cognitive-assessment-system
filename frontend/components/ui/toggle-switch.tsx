"use client";

import { useState } from "react";

interface ToggleSwitchProps {
  label: string;
  defaultChecked?: boolean;
  onChange?: (checked: boolean) => void;
}

export default function ToggleSwitch({ label, defaultChecked = false, onChange }: ToggleSwitchProps) {
  const [checked, setChecked] = useState(defaultChecked);

  const handleChange = () => {
    const newValue = !checked;
    setChecked(newValue);
    onChange?.(newValue);
  };

  return (
    <div className="flex items-center justify-between w-full">
      <span className="text-gray-700 font-medium">{label}</span>
      <button
        onClick={handleChange}
        className={`w-12 h-6 flex items-center rounded-full p-1 transition-colors ${
          checked ? "bg-teal-500" : "bg-gray-300"
        }`}
      >
        <div
          className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
            checked ? "translate-x-6" : "translate-x-0"
          }`}
        />
      </button>
    </div>
  );
}
