"use client";

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

interface MMSETrendChartProps {
  data: Array<{ date: string; mmse: number; session: string }>;
}

export function MMSETrendChart({ data }: MMSETrendChartProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-amber-900">{`Ngày: ${label}`}</p>
          <p className="text-amber-800">{`Điểm MMSE: ${data.mmse}`}</p>
          <p className="text-sm text-gray-600">{`Session: ${data.session}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f59e0b" opacity={0.3} />
        <XAxis
          dataKey="date"
          stroke="#92400e"
          fontSize={12}
          tick={{ fill: '#92400e' }}
        />
        <YAxis
          domain={[0, 30]}
          stroke="#92400e"
          fontSize={12}
          tick={{ fill: '#92400e' }}
          label={{ value: 'Điểm MMSE', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#92400e' } }}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={24} stroke="#dc2626" strokeDasharray="5 5" label={{ value: "Ngưỡng bình thường", position: "topRight" as any, fill: "#dc2626" }} />
        <ReferenceLine y={18} stroke="#ea580c" strokeDasharray="5 5" label={{ value: "MCI", position: "topRight" as any, fill: "#ea580c" }} />
        <Line
          type="monotone"
          dataKey="mmse"
          stroke="#92400e"
          strokeWidth={3}
          dot={{ fill: '#f59e0b', strokeWidth: 2, r: 6 }}
          activeDot={{ r: 8, stroke: '#92400e', strokeWidth: 2, fill: '#fbbf24' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
