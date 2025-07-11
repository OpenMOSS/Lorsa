import { cn } from "@/lib/utils";
import { Link, useLocation } from "react-router-dom";

export const AppNavbar = () => {
  const location = useLocation();

  return (
    <nav className="p-4">
      <div className="container mx-auto flex items-center gap-8">
        <img src="/openmoss.ico" alt="logo" className="h-8" />

        <div className="flex gap-4 items-center">
          <Link
            className={cn(
              "transition-colors hover:text-foreground/80 text-foreground/60",
              location.pathname === "/lorsa" && "text-foreground"
            )}
            to="/lorsa"
          >
            Lorsa Visualizer
          </Link>
        </div>
      </div>
    </nav>
  );
};
