import Link from "next/link";
import { Button } from "@/components/ui/button";
import Image from "next/image";

type Props = {
  activeCourse: {
    imageSrc: string;
    title: string;
  };
  hearts: number;
  points: number;
};

export const UserProgress = (props: Props) => {
  const { activeCourse, hearts, points } = props;

  return (
    <div className="flex items-center justify-between gap-x-4 w-full mt-4 p-4 bg-white rounded-2xl border border-slate-200 shadow-sm">
      <Link href="/menu">
        <Button 
          variant="ghost" 
          className="p-2 hover:bg-slate-100 transition-colors duration-200 rounded-xl"
        >
          <Image
            src={activeCourse.imageSrc}
            alt={activeCourse.title}
            className="rounded-xl border-2 border-slate-200 shadow-sm"
            width={56}
            height={56}
          />
        </Button>
      </Link>

      <div className="flex items-center gap-x-6">
        <Link href="/info">
          <Button 
            variant="ghost" 
            className="text-orange-500 hover:text-orange-600 flex items-center gap-x-2 px-4 py-2 rounded-xl hover:bg-orange-50 transition-all duration-200 font-semibold"
          >
            <div className="bg-orange-100 p-2 rounded-lg">
              <Image src="/points.svg" height={24} width={24} alt="Points" />
            </div>
            <span className="text-lg font-bold">{points}</span>
          </Button>
        </Link>

        <Link href="/stats">
          <Button 
            variant="ghost" 
            className="text-rose-500 hover:text-rose-600 flex items-center gap-x-2 px-4 py-2 rounded-xl hover:bg-rose-50 transition-all duration-200 font-semibold"
          >
            <div className="bg-rose-100 p-2 rounded-lg">
              <Image src="/heart.svg" height={24} width={24} alt="Hearts" />
            </div>
            <span className="text-lg font-bold">{hearts}</span>
          </Button>
        </Link>
      </div>
    </div>
  );
};
