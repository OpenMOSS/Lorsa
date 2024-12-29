import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

export const RootPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    navigate("/lorsa" + location.search, { replace: true });
  }, [location, navigate]);

  return null;
};
