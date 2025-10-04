"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface UserData {
	name: string;
	age: string;
	gender: string;
	email: string;
	phone: string;
	title?: string;
}

interface PersonalInfoFormProps {
	onSubmitSuccess?: () => void;
}

export default function PersonalInfoForm({ onSubmitSuccess }: PersonalInfoFormProps) {
	const [form, setForm] = useState<UserData>({
		name: "",
		age: "",
		gender: "",
		email: "",
		phone: "",
		title: ""
	});
	const [errors, setErrors] = useState<Record<string, string>>({});
	const [isSubmitting, setIsSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState<string | null>(null);

	const validate = () => {
		const e: Record<string, string> = {};
		if (!form.name.trim()) e.name = "Vui lòng nhập họ và tên";
		if (!form.age || isNaN(Number(form.age))) e.age = "Tuổi không hợp lệ";
		if (!form.gender) e.gender = "Vui lòng chọn giới tính";
		if (!form.email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) e.email = "Email không hợp lệ";
		if (!form.phone) e.phone = "Vui lòng nhập số điện thoại";
		setErrors(e);
		return Object.keys(e).length === 0;
	};

	const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
		const { name, value } = e.target;
		setForm((prev) => ({ ...prev, [name]: value }));
	};

	async function saveProfileToDb() {
		// New DB-backed API
		const res = await fetch("/api/profile/user", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(form)
		});
		if (!res.ok) {
			throw new Error(await res.text());
		}
		return res.json();
	}

	const handleSubmit = async () => {
		setSubmitError(null);
		if (!validate()) return;
		setIsSubmitting(true);
		try {
			// Save to DB
			await saveProfileToDb();
			// Fallback legacy API (kept for compatibility, ignore result)
			fetch("/api/profile", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(form)
			}).catch(() => {});

			// Send email confirmation (non-blocking)
			fetch("/api/send-email", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ userData: form })
			}).catch(() => {});

			// Persist local flag
			try { localStorage.setItem("profileCompleted", "true"); } catch {}

			onSubmitSuccess?.();
		} catch (err: any) {
			setSubmitError("Không thể lưu thông tin. Vui lòng thử lại.");
		} finally {
			setIsSubmitting(false);
		}
	};

	return (
		<Card className="p-4 bg-white/90 max-w-xl mx-auto">
			<h3 className="text-lg font-semibold mb-3">Thông tin cá nhân</h3>
			<div className="grid grid-cols-1 md:grid-cols-2 gap-3">
				<div>
					<label className="text-sm">Họ và tên</label>
					<input name="name" value={form.name} onChange={handleChange} className="w-full border rounded px-3 py-2" />
					{errors.name && <p className="text-xs text-red-600 mt-1">{errors.name}</p>}
				</div>
				<div>
					<label className="text-sm">Tuổi</label>
					<input name="age" value={form.age} onChange={handleChange} className="w-full border rounded px-3 py-2" />
					{errors.age && <p className="text-xs text-red-600 mt-1">{errors.age}</p>}
				</div>
				<div>
					<label className="text-sm">Giới tính</label>
					<select name="gender" value={form.gender} onChange={handleChange} className="w-full border rounded px-3 py-2">
						<option value="">Chọn</option>
						<option value="Nam">Nam</option>
						<option value="Nữ">Nữ</option>
						<option value="Khác">Khác</option>
					</select>
					{errors.gender && <p className="text-xs text-red-600 mt-1">{errors.gender}</p>}
				</div>
				<div>
					<label className="text-sm">Email</label>
					<input name="email" value={form.email} onChange={handleChange} className="w-full border rounded px-3 py-2" />
					{errors.email && <p className="text-xs text-red-600 mt-1">{errors.email}</p>}
				</div>
				<div>
					<label className="text-sm">Số điện thoại</label>
					<input name="phone" value={form.phone} onChange={handleChange} className="w-full border rounded px-3 py-2" />
					{errors.phone && <p className="text-xs text-red-600 mt-1">{errors.phone}</p>}
				</div>
				<div>
					<label className="text-sm">Chức danh (tuỳ chọn)</label>
					<input name="title" value={form.title} onChange={handleChange} className="w-full border rounded px-3 py-2" />
				</div>
			</div>
			{submitError && <p className="text-sm text-red-600 mt-3">{submitError}</p>}
			<div className="mt-4 flex gap-2">
				<Button onClick={handleSubmit} disabled={isSubmitting} className="px-4">
					{isSubmitting ? "Đang lưu..." : "Lưu & Tiếp tục"}
				</Button>
			</div>
		</Card>
	);
}
