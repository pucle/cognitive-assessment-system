import {Button} from "@/components/ui/button"
const ButtonsPage = () => {
    return (
        <div className="p-4 space-y-4 flex flex-col max-w-[200px]">
  <Button className="active:translate-y-1 active:shadow-inner transition-all">
    DEFAULT
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="primary" >
    PRIMARY
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="primaryOutline">
    PRIMARY OUTLINE
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="secondary" >
    Secondary
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="secondaryOutline">
    secondary OUTLINE
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="danger" >
    danger
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="dangerOutline">
    danger OUTLINE
  </Button>
    <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="super" >
    super
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="superOutline">
    super OUTLINE
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="ghost">
    ghost
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="sidebar" >
    sidebar
  </Button>
  <Button className="active:translate-y-1 active:shadow-inner transition-all" variant="sidebarOutline">
    sidebar OUTLINE
  </Button>
  
  
  
  
  
</div>

    );
};

export default ButtonsPage;
