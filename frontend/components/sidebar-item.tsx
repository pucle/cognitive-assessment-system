"use client";

import { usePathname } from "next/navigation";
import { Button } from "./ui/button";
import Link from "next/link";
import Image from "next/image";

type Props = {
    label: string;
    iconSrc: string;
    href: string;
};
export const SidebarItem = ({ label, iconSrc, href }: Props) => {
    const pathname = usePathname();
    const active = pathname === href;

    // Hide game page completely
    if (href === '/game') {
        return null;
    }
    return (
        <Button variant={active ? "sidebarOutline" : "sidebar"} className={`justify-start h-[46px] text-[13px] ${active ? 'bg-amber-400/10 text-orange-900 border-amber-300' : 'hover:bg-amber-50/60 text-orange-800'} rounded-2xl`} asChild>
            <Link href={href}>
                <Image src={iconSrc} alt={label} width={28} height={28} className=" rounded-md" />
                {label}
            </Link>
            
        </Button>
    );
};