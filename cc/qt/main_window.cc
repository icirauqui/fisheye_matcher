#include "cc/qt/main_window.h"

#include "cc/qt/ui_main_window.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui_(new Ui::MainWindow)
{
  ui_->setupUi(this);
}

MainWindow::~MainWindow() {}