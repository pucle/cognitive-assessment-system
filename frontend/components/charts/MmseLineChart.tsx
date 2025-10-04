'use client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { format } from 'date-fns';

interface MmseChartData {
  date: string;
  mmseScore: number;
  gptScore: number;
  sessionId: string;
  displayDate: string;
}

interface MmseLineChartProps {
  data: any[];
  className?: string;
}

export default function MmseLineChart({ data, className = '' }: MmseLineChartProps) {
  // Transform data for chart
  const chartData: MmseChartData[] = data
    .filter(item => item.finalMmseScore !== null && item.finalMmseScore !== undefined)
    .map(item => {
      const date = new Date(item.createdAt);
      return {
        date: date.toISOString(),
        mmseScore: item.finalMmseScore || 0,
        gptScore: item.overallGptScore || 0,
        sessionId: item.sessionId,
        displayDate: format(date, 'dd/MM/yyyy')
      };
    })
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  if (chartData.length === 0) {
    return (
      <div className={`bg-white p-6 rounded-xl shadow-lg ${className}`}>
        <h3 className="text-lg font-bold mb-4">📈 Biến Động Điểm MMSE</h3>
        <div className="text-center py-8 text-gray-500">
          <div className="text-4xl mb-2">📊</div>
          <p>Chưa có đủ dữ liệu để hiển thị biểu đồ</p>
          <p className="text-sm">Cần ít nhất 1 kết quả có điểm MMSE</p>
        </div>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="font-semibold">{data.displayDate}</p>
          <p className="text-blue-600">MMSE: <span className="font-bold">{payload[0].value}/30</span></p>
          <p className="text-purple-600">GPT Score: <span className="font-bold">{payload[1]?.value}/10</span></p>
          <p className="text-xs text-gray-500">ID: {data.sessionId}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className={`bg-white p-6 rounded-xl shadow-lg ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold">📈 Biến Động Điểm MMSE Theo Thời Gian</h3>
        <div className="text-sm text-gray-500">
          {chartData.length} điểm dữ liệu
        </div>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="displayDate"
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis
              domain={[0, 30]}
              tick={{ fontSize: 12 }}
              label={{ value: 'Điểm MMSE', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="mmseScore"
              stroke="#3B82F6"
              strokeWidth={3}
              dot={{ fill: '#3B82F6', strokeWidth: 2, r: 6 }}
              activeDot={{ r: 8 }}
              name="Điểm MMSE"
            />
            <Line
              type="monotone"
              dataKey="gptScore"
              stroke="#8B5CF6"
              strokeWidth={2}
              dot={{ fill: '#8B5CF6', strokeWidth: 2, r: 4 }}
              name="GPT Score"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Chart Summary */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
        <div className="bg-blue-50 p-3 rounded">
          <div className="font-bold text-blue-600">{Math.max(...chartData.map(d => d.mmseScore))}</div>
          <div className="text-xs text-gray-600">Cao nhất</div>
        </div>
        <div className="bg-red-50 p-3 rounded">
          <div className="font-bold text-red-600">{Math.min(...chartData.map(d => d.mmseScore))}</div>
          <div className="text-xs text-gray-600">Thấp nhất</div>
        </div>
        <div className="bg-green-50 p-3 rounded">
          <div className="font-bold text-green-600">
            {(chartData.reduce((sum, d) => sum + d.mmseScore, 0) / chartData.length).toFixed(2)}
          </div>
          <div className="text-xs text-gray-600">Trung bình</div>
        </div>
        <div className="bg-purple-50 p-3 rounded">
          <div className="font-bold text-purple-600">
            {chartData.length > 1 ?
              (chartData[chartData.length - 1].mmseScore - chartData[0].mmseScore > 0 ? '↗️' : '↘️')
              : '➖'
            }
          </div>
          <div className="text-xs text-gray-600">Xu hướng</div>
        </div>
      </div>
    </div>
  );
}
